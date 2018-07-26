// 边框数据层====================
// 输入数据形式=========================
// Datum datum;
// 设置图片 data 像素数据区域============
//    datum->set_data()
//    datum->set_encoded(true);// 编码标志
// 标签域================================
//    datum->add_float_data(float(label));    //label  标签
//    datum->add_float_data(float(difficult));//diff   偏移?
//    datum->add_float_data(box[i]);//box[4]  x,y,w,h  (范围0~1)
///////////////////////////
// 输出 top的形式=====================================
// top[0] 为 数据区 data 
// top_data = batch->data_.mutable_cpu_data();
// top[0]->num()       图片数量 N
// top[0]->channels()  通道数量 3
// top[0]->height()    尺寸 300/416
// top[0]->width();

// =====================================================
// top[1] 为 标签区 label 需要以不同格子大小区分()=======13*13格子======
// top_label ： batch->multi_label_[i]->mutable_cpu_data() 
//   N*150 ======
//  150 = 30*(1+4)  预设 30个边框空间，5个位一组边框参数，边框数量不足30， 后面填0============
// top[2]======================================26*26格子============


#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/box_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {
	
//          reader_ (读数据到队列 data_reader ) ->
//  (prefetch 预读取 batch的数量    batch一次读取图像数量) 
//  (shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));// 数据库对象)
//  (db->Open(param_.data_param().source(), db::READ);// 打开数据库文件)
// BaseDataLayer  ->  BasePrefetchingDataLayer  -> BoxDataLayer 
// prefetch_free_(), prefetch_full_(), InternalThread 多线程读取数据
// transform_param_(数据转换，去均值处理)  -> BaseDataLayer
// 类构造函数=================================
template <typename Dtype>
BoxDataLayer<Dtype>::BoxDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),// 多线程读取+数据预处理
    reader_(param) // (读数据到队列 data_reader.cpp ) 
{
}
// 类析构函数=================================
template <typename Dtype>
BoxDataLayer<Dtype>::~BoxDataLayer() {
  this->StopInternalThread();
}

// BoxDataLayer 层初始化================
template <typename Dtype>
void BoxDataLayer<Dtype>::DataLayerSetUp(
const vector<Blob<Dtype>*>& bottom,// 这里底部没有数据过来，直接从数据库读取
const vector<Blob<Dtype>*>& top)   // 输出数据和标签
{
  this->box_label_ = true;
  // 数据层参数
  const DataParameter param = this->layer_param_.data_param();
  
  const int batch_size = param.batch_size();
  // batch 参数 一次载入数据大小(图像数量)====================
  // Read a data point, and use it to initialize the top blob.
  // 从 队列里面读取 一个 Datum 数据节点 
  Datum& datum = *(reader_.full().peek());
  
////////////////////////////////////////////////////////////////////
// data 数据域 尺寸 N*3*300*300 / N*3*416*416 等
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
	  
///////////////////////////////////////////////////////////////
// label 标签域 尺寸
  if (this->output_labels_) 
  {
    if (param.side_size() > 0) // 最后 特征图尺寸 7*7 / 13*13等
	{
      for (int i = 0; i < param.side_size(); ++i) 
	  {
        sides_.push_back(param.side(i));
      }
    }
    if (sides_.size() == 0) 
	{
      sides_.push_back(7);// 默认 7*7  13*13 ....不同格子尺寸对应不同 yolo版本的输出标签
    }
	
	//////////////  top输出维度 和 格子维度一致???
    CHECK_EQ(sides_.size(), top.size() - 1) << "side num not equal to top size";
     
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
	{
      this->prefetch_[i].multi_label_.clear(); // 预数据 清理
    }
	
    for (int i = 0; i < sides_.size(); ++i) {// 1个 13  26   39 ... 
		
      vector<int> label_shape(1, batch_size);// N*150
//******************************************************************************//
//int label_size = sides_[i] * sides_[i] * (1 + 1 + 1 + 4); //side_*side*(obj,cls_label_,box_[4])

      int label_size = (30 * 5); //(maxboxes=30)*(4+1)  最多30个边框 每个边框 类别id + box_[4] 
      label_shape.push_back(label_size);
      top[i+1]->Reshape(label_shape);// 不同格子数量 对应不同缩放尺寸的表桥 但是长度都设置为 30*5
      for (int j = 0; j < this->PREFETCH_COUNT; ++j) 
	  {
        shared_ptr<Blob<Dtype> > tmp_blob;
        tmp_blob.reset(new Blob<Dtype>(label_shape));// N*150
        this->prefetch_[j].multi_label_.push_back(tmp_blob);
      }
    }
  }
}

// 从数据库 队列内读取 合适的数据到 batch
// This function is called on prefetch thread
template<typename Dtype>
void BoxDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  
  const int batch_size = this->layer_param_.data_param().batch_size();// 批次大小
  Datum& datum = *(reader_.full().peek());// 读取队列内的 datum 数据
  
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;      //  批次大小图片数量
  batch->data_.Reshape(top_shape);//  图像数据域=======================

  Dtype* top_data = batch->data_.mutable_cpu_data();
  
  vector<Dtype*> top_label;

  
  if (this->output_labels_) {
	  /////////////////////   
    for (int i = 0; i < sides_.size(); ++i) {
      top_label.push_back(batch->multi_label_[i]->mutable_cpu_data());
    }
  }
  /////////////////////// 处理每张图片
  for (int item_id = 0; item_id < batch_size; ++item_id) {
	  
    timer.Start();
    // get a datum
// 从队列里 读取一个数据=======================================
    Datum& datum = *(reader_.full().pop("Waiting for data from queue..."));
    read_time += timer.MicroSeconds();
    timer.Start();
	
	
// 图片镜像、缩放、剪裁等===========================================
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
	
//////////////////////////////////////
    vector<BoxLabel> box_labels;
/////////////////////////////////////////

    if (this->output_labels_) {
      //*****************************************************************//
	  
 // 变换图像和标签================================
///////////////////////////data_transformer_ 需要配修改==================
      // rand sample a patch, adjust box labels
      this->data_transformer_->Transform(datum, 
	                                     &(this->transformed_data_), 
										 &box_labels);// 获取box_labels 数据

// 变换标签==============================================
      // transform label
      for (int i = 0; i < sides_.size(); ++i) 
	  {
        int label_offset = batch->multi_label_[i]->offset(item_id);
        int count  = batch->multi_label_[i]->count(1);// 30*5 
        //LOG(INFO) << "sides_.size: " << sides_.size() << "label_offset:" << label_offset << "count:" << count;
        transform_label(count, top_label[i] + label_offset, box_labels, sides_[i]);
      }
    } 
	else 
	{// 转换数据==========================
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
	
//  内存清理方式 =================================
    trans_time += timer.MicroSeconds();
	
    reader_.free().push(const_cast<Datum*>(&datum));
  }
  
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  
}



// 变换标签=============================================================
template<typename Dtype>
void BoxDataLayer<Dtype>::transform_label(
int count, 
Dtype* top_label,
const vector<BoxLabel>& box_labels, // 该张图片 的 标记边框
int side) 
{
  //int locations = pow(side, 2);
  CHECK_EQ(count, 30*5) << "side and count not match";// 30*( 1+4 , 类别id + 边框box[4])
  // label
  caffe_set(30*5, Dtype(0), top_label);// 输出top_label 设置为0
  // isobj
  //caffe_set(locations, Dtype(0), top_label + locations);
  // class label
  //caffe_set(locations, Dtype(-1), top_label + locations * 2);
  // box
  //caffe_set(locations*4, Dtype(0), top_label + locations * 3);
  int index = 0;
  for (int i = 0; i < box_labels.size(); ++i) 
  {
    float difficult = box_labels[i].difficult_;
    if (difficult != 0. && difficult != 1.) 
	{
      LOG(WARNING) << "Difficult must be 0 or 1";
    }
	// 类别标签
    float class_label = box_labels[i].class_label_; //box_labels[i]: BoxLabel
    CHECK_GE(class_label, 0) << "class_label must >= 0";
    //float x = box_labels[i].box_[0];
    //float y = box_labels[i].box_[1];
    
    //int x_index = floor(x * side);
    //int y_index = floor(y * side);
    //x_index = std::min(x_index, side - 1);
    //y_index = std::min(y_index, side - 1);
    //int dif_index = side * y_index + x_index; //
    //int obj_index = locations + dif_index; 
    //int class_index = locations * 2 + dif_index;
    //int cor_index = locations * 3 + dif_index * 4;
	
    top_label[index++] = class_label;// 类别id ===================================
	//top_label[index++] = 1;
    
    //top_label[obj_index] = 1;
    // LOG(INFO) << "dif_index: " << dif_index << " class_label: " << class_label;
    //top_label[class_index] = class_label;
    for (int j = 0; j < 4; ++j) 
	{
	top_label[index + j] = box_labels[i].box_[j];// 边框box[4]===================================
	
        //top_label[index + j] = top_label[index + j]<1.0?top_label[index+j]:1.0;
	//top_label[index + j] = top_label[index + j]>0.0?top_label[index+j]:0.0;
	// top_label[index + j] = 1;
      //LOG(INFO) << "box_: " << box_labels[i].box_[j];
    }
    index += 4;// 跳转到下一个标签域
    //LOG(INFO) <<"index: " <<index <<" label: " <<class_label << " x: " << x << " y: " << y;
  }
}

INSTANTIATE_CLASS(BoxDataLayer);
REGISTER_LAYER_CLASS(BoxData);

}  // namespace caffe
