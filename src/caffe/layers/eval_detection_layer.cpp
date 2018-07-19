#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/detection_loss_layer.hpp"
#include "caffe/layers/eval_detection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

class BoxData {
 public:
  int label_;
  float score_;
  vector<float> box_;
};

inline float sigmoid(float x)
{
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
float softmax_region(Dtype* input, int classes)
{
  Dtype sum = 0;
  Dtype large = input[0];

  for (int i = 0; i < classes; ++i){
    if (input[i] > large)
      large = input[i];
  }
  for (int i = 0; i < classes; ++i){
    Dtype e = exp(input[i] - large);
    sum += e;
    input[i] = e;
  }
  for (int i = 0; i < classes; ++i){
    input[i] = input[i] / sum;
  }
  return 0;
}

bool BoxSortDecendScore(const BoxData& box1, const BoxData& box2) {
  return box1.score_ > box2.score_;
}

void ApplyNms(const vector<BoxData>& boxes, vector<int>* idxes, float threshold) {
  map<int, int> idx_map;
  for (int i = 0; i < boxes.size() - 1; ++i) {
    if (idx_map.find(i) != idx_map.end()) {
      continue;
    }
    vector<float> box1 = boxes[i].box_;
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (idx_map.find(j) != idx_map.end()) {
        continue;
      }
      vector<float> box2 = boxes[j].box_;
      float iou = Calc_iou(box1, box2);
      if (iou >= threshold) {
        idx_map[j] = 1;
      }
    }
  }
  for (int i = 0; i < boxes.size(); ++i) {
    if (idx_map.find(i) == idx_map.end()) {
      idxes->push_back(i);
    }
  }
}
// 返回一张图像的 标签 + 多边框数据的 标签
template <typename Dtype>
void GetGTBox(int side, const Dtype* label_data, map<int, vector<BoxData> >* gt_boxes) {
  // 输入数据指针数组  label_data 以及确定对应图片的起始地址
  // 数据形式  30*5  一张图片 30个物体边框 第一个为标签 后面的为 边框 x y w h
  //int locations = pow(side, 2);
  //for (int i = 0; i < locations; ++i) {
  for (int i = 0; i < 30; ++ i) {// 30个物体边框
    //if (!label_data[locations + i]) {
    if (label_data[i * 5 + 1] == 0) {// 有的图片标注框没有30个，其余设置为0
	//continue;
	break; //maybe problem??? 开始出现0了，后面应该全部是0 跳过全部
    }
    BoxData gt_box;
    //bool difficult = (label_data[i] == 1);
    //int label = static_cast<int>(label_data[locations * 2 + i]);
    int label = label_data[i * 5 + 0];// 标签
    //gt_box.difficult_ = difficult;
    gt_box.label_ = label;
	
//////////////////////////////////////////////////////////////////////
///////////////////////////////////////////
    // gt_box.score_ = 1;
    gt_box.score_ = i; // 真实边框 的得分应该设置为1  
    //int box_index = locations * 3 + i * 4;
    int box_index = i * 5 + 1;// 边框起始指针 x y w h
    //LOG(INFO) << "label:" << label;
    for (int j = 0; j < 4; ++j) {
      gt_box.box_.push_back(label_data[box_index + j]);
      //LOG(INFO) << "x,y,w,h:" << label_data[box_index + j];
    }
    if (gt_boxes->find(label) == gt_boxes->end()) {
      (*gt_boxes)[label] = vector<BoxData>(1, gt_box);// 一类中唯一一个物体
    } else {
      (*gt_boxes)[label].push_back(gt_box);// 一类中多个物体
    }
  }
}

// 获取预测边框 =============================================
template <typename Dtype>
void GetPredBox(int side, int num_object,  //  格子 13  5个物体  20种类别
               int num_class, Dtype* input_data,  // 输入数据 13*13*125 -> 13*13*5*25
               map<int, vector<BoxData> >* pred_boxes, //  边框   评分类别  nms阈值(重叠)
  int score_type, float nms_threshold, vector<Dtype> biases) {

 vector<BoxData> tmp_boxes;// 筛选出来的
  //int locations = pow(side, 2);
  for (int j = 0; j < side; ++j)// 13    0->12 格子
    for (int i = 0; i < side; ++i)// 13  0->12 
      for (int n = 0; n < 5; ++n)// 5    0->4  物体数量
      {// 25 = 4边框参数 + 1置信度 + 20种类别概率  
	  
	// 起始指针
	int index = (j * side + i) * num_object * (num_class + 1 + 4) + n * (num_class + 1 + 4);
	
	// 坐标中心 sigmoid激活得到格子偏移量
	float x = (i + sigmoid(input_data[index + 0])) / side;// 格子偏移量/总格子数量 归一化到 0~1之间
	float y = (j + sigmoid(input_data[index + 1])) / side;
	
	// 边框长度指数映射之后 分别被五种边框尺寸系数加权之后 除以/总格子数量 归一化到 0~1之间
 	float w = (exp(input_data[index + 2]) * biases[2 * n]) / side;
	float h = (exp(input_data[index + 3]) * biases[2 * n + 1]) / side;
	
    // 置信度 在后面处理 需要 sigmoid() 处理到0~1=======================
	
    // 从20种预测概率种选出概率最大的，作为本边框的预测 类比=================
	// 20种 类别预测概率处理
	softmax_region(input_data + index + 5, num_class);
	int pred_label = 0;
	// 在20种 类别预测概率 种选出 概率最大的 
	float max_prob = input_data[index + 5];// 初始化第一种物体的预测概率为最大
	
	for (int c = 0; c < num_class; ++c)
	{
	  if (max_prob < input_data[index + 5 + c])
	  {
	    max_prob = input_data[index + 5 + c];// 记录预测概率 最大的值
	    pred_label = c; // 0,..,19(20类)     对应的类别 标签
	  }
	}
	BoxData pred_box;
	pred_box.label_ = pred_label;// 预测类别标签
	
 // 置信度 需要 sigmoid() 处理到0~1===========================================
    float obj_score = sigmoid(input_data[index + 4]);
	
	if (score_type == 0) {
	  pred_box.score_ = obj_score;// 按照 置信度 进行评价
	} 
	else if (score_type == 1) 
	{
	  pred_box.score_ = max_prob; // 按照 类别预测概率进行评价
	} 
	else {
	  pred_box.score_ = obj_score * max_prob;// 按照 置信度*类别预测概率 进行评价
	}

	pred_box.box_.push_back(x);
	pred_box.box_.push_back(y);
   	pred_box.box_.push_back(w);
	pred_box.box_.push_back(h);
	
	tmp_boxes.push_back(pred_box);
	//LOG(INFO)<<"Not nms pred_box:" << pred_box.label_ << " " << obj_score << " " << max_prob  << " " << pred_box.score_ << " " << pred_box.box_[0] << " " << pred_box.box_[1] << " " << pred_box.box_[2] << " " << pred_box.box_[3];	
    }  
  /*
  for (int i = 0; i < locations; ++i) {
    int pred_label = 0;
    float max_prob = input_data[i];
    for (int j = 1; j < num_class; ++j) {
      int class_index = j * locations + i;   
      if (input_data[class_index] > max_prob) {
        pred_label = j;
        max_prob = input_data[class_index];
      }
    }
    if (nms_threshold < 0) {
      if (pred_boxes->find(pred_label) == pred_boxes->end()) {
        (*pred_boxes)[pred_label] = vector<BoxData>();
      }
    }
    // LOG(INFO) << "pred_label: " << pred_label << " max_prob: " << max_prob; 
    int obj_index = num_class * locations + i;
    int coord_index = (num_class + num_object) * locations + i;
    for (int k = 0; k < num_object; ++k) {
      BoxData pred_box;
      float scale = input_data[obj_index + k * locations];
      pred_box.label_ = pred_label;
      if (score_type == 0) {
        pred_box.score_ = scale;
      } else if (score_type == 1) {
        pred_box.score_ = max_prob;
      } else {
        pred_box.score_ = scale * max_prob;
      }
      int box_index = coord_index + k * 4 * locations;
      if (!constriant) {
        pred_box.box_.push_back(input_data[box_index + 0 * locations]);
        pred_box.box_.push_back(input_data[box_index + 1 * locations]);
      } else {
        pred_box.box_.push_back((i % side + input_data[box_index + 0 * locations]) / side);
        pred_box.box_.push_back((i / side + input_data[box_index + 1 * locations]) / side);
      }
      float w = input_data[box_index + 2 * locations];
      float h = input_data[box_index + 3 * locations];
      if (use_sqrt) {
        pred_box.box_.push_back(pow(w, 2));
        pred_box.box_.push_back(pow(h, 2));
      } else {
        pred_box.box_.push_back(w);
        pred_box.box_.push_back(h);
      }
      if (nms_threshold >= 0) {
        tmp_boxes.push_back(pred_box);
      } else {
        (*pred_boxes)[pred_label].push_back(pred_box);
      }
    }
  }*/
  if (nms_threshold >= 0) {
    std::sort(tmp_boxes.begin(), tmp_boxes.end(), BoxSortDecendScore);
    vector<int> idxes;
    ApplyNms(tmp_boxes, &idxes, nms_threshold);
    for (int i = 0; i < idxes.size(); ++i) {
      BoxData box_data = tmp_boxes[idxes[i]];
      //**************************************************************************************//
      if (box_data.score_ < 0.005) // from darknet
	continue;
      //LOG(INFO)<<"box_data:" << box_data.label_ << " " << box_data.score_ << " " << box_data.box_[0] << " " << box_data.box_[1] << " " << box_data.box_[2] << " " << box_data.box_[3];
      if (pred_boxes->find(box_data.label_) == pred_boxes->end()) {
        (*pred_boxes)[box_data.label_] = vector<BoxData>();
      }
      (*pred_boxes)[box_data.label_].push_back(box_data);
    }
  } else {
    for (std::map<int, vector<BoxData> >::iterator it = pred_boxes->begin(); it != pred_boxes->end(); ++it) {
      std::sort(it->second.begin(), it->second.end(), BoxSortDecendScore);
    }
  }
}
// 层 初始化
template <typename Dtype>
void EvalDetectionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  EvalDetectionParameter param = this->layer_param_.eval_detection_param();
  side_ = param.side();//13*13格子
  num_class_ = param.num_class();// voc 20类 / coco 80类
  num_object_ = param.num_object();// 一个格子预测5个边框
  threshold_ = param.threshold();
  //sqrt_ = param.sqrt();
  //constriant_ = param.constriant();
  
  nms_ = param.nms();
  
  for (int c = 0; c < param.biases_size(); ++c){// 5种边框尺寸 10个参数
    biases_.push_back(param.biases(c));
  }

  switch (param.score_type()) {
    case EvalDetectionParameter_ScoreType_OBJ:
      score_type_ = 0;
      break;
    case EvalDetectionParameter_ScoreType_PROB:
      score_type_ = 1;
      break;
    case EvalDetectionParameter_ScoreType_MULTIPLY:
      score_type_ = 2;
      break;
    default:
      LOG(FATAL) << "Unknow score type.";
  }
}

template <typename Dtype>
void EvalDetectionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int input_count = bottom[0]->count(1); //b*13*13*125 网络输出
  int label_count = bottom[1]->count(1); //b*30*5      标签 预设30个物体 4边框+1类别
  // outputs: classes, iou, coordinates
  //int tmp_input_count = side_ * side_ * (num_class_ + (1 + 4) * num_object_);
  int tmp_input_count = side_ * side_ * num_object_ *( num_class_ + 4 + 1 ); //13*13*5*25
  // label: isobj, class_label, coordinates
  //int tmp_label_count = side_ * side_ * (1 + 1 + 1 + 4);
  int tmp_label_count = 30 * 5;// 

  CHECK_EQ(input_count, tmp_input_count);// 确保网络输出 每张图片为 13*13*5*25 大小 
                                         // 13*13个格子，每个格子预测5种边框,每种边框预测 20类概率+4边框参数+1置信度
  CHECK_EQ(label_count, tmp_label_count);// 而标签输入   每张图片为 30*5       大小 30个物体边框，4个边框参数+1个类别标签

  vector<int> top_shape(2, 1);// 两行一列
  top_shape[0] = bottom[0]->num();// 图片数量
  //top_shape[1] = num_class_ + side_ * side_ * num_object_ * 4; 
  // 20 + 13*13*5*4 标签 得分 TP FP 
  top_shape[1] = num_class_ + side_ * side_ * num_object_ * 4; //13*13*5*(label + score + tp + fp)
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void EvalDetectionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //const Dtype* input_data = bottom[0]->cpu_data();// 网络输出     N*13*13*125
  const Dtype* label_data = bottom[1]->cpu_data();  // 真实标签数据 N*30*5
  //LOG(INFO) << bottom[0]->data_at(0,0,0,0) << " " << bottom[0]->data_at(0,0,0,1);  
  
  Blob<Dtype> swap;// 网络输出数据 N * 13* 13* 125
  // 变形为    N * (13 * 13) *  5 * 25
  swap.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_object_, bottom[0]->channels()/num_object_);  
  
  Dtype* swap_data = swap.mutable_cpu_data();// cpu上的数据
  int index = 0;
  for (int b = 0; b < bottom[0]->num(); ++b)// 图片数量
    for (int h = 0; h < bottom[0]->height(); ++h) // 格子 13
      for (int w = 0; w < bottom[0]->width(); ++w)// 格子 13
        for (int c = 0; c < bottom[0]->channels(); ++c)// 5*25=125
	{
	  swap_data[index++] = bottom[0]->data_at(b,c,h,w);
	}  
  //*******************************************************************************//
  //caffe_set(swap.count(), Dtype(0.0), swap_data);
  //int p_index = (7*13+4)*125;
  //swap_data[p_index]=-0.1020;
  //swap_data[p_index+1]=2.0867;
  //swap_data[p_index+2]=1.612;
  //swap_data[p_index+3]=1.0515;
  //swap_data[p_index+4]=1.0;
  //swap_data[p_index+5+11]=100;  

  //*******************************************************************************//  
  Dtype* top_data = top[0]->mutable_cpu_data();// 层输出 cpu数据
  caffe_set(top[0]->count(), Dtype(0), top_data);
  
  for (int i = 0; i < bottom[0]->num(); ++i) {// N  图片数量
    int input_index = i * bottom[0]->count(1);// 网络输出标签 i * 13*13*125
    int true_index = i * bottom[1]->count(1);//  真实标签     i * 30*5
    int top_index = i * top[0]->count(1);    //  输出数据     i * ( 20 + 13*13*5*4)
                                             //  前面20个为 真实标签 物体类别出现的次数
 
 // 获取真实边框 =========================================
    map<int, vector<BoxData> > gt_boxes;
    // 从 对应图片的标签数据中 获取 真实边框 label_ + score_ + box_ 
    // 返回一张图像的 标签 + 多边框数据的 标签
    GetGTBox(side_, label_data + true_index, &gt_boxes);

 
// 在输出数据中  记录 真实标签 物体类别出现的次数=======================
    for (std::map<int, vector<BoxData > >::iterator it = gt_boxes.begin(); it != gt_boxes.end(); ++it) {
      // 遍历 每一个 真实的标签
      int label = it->first;// 标签 类别
      vector<BoxData>& g_boxes = it->second;// BoxData: label_ + score_ + box_ 
      for (int j = 0; j < g_boxes.size(); ++j) {// 边框数量
          top_data[top_index + label] += 1;     // 真实标签 物体类别出现的次数
      }
    }
// 获取预测边框 =============================================
    map<int, vector<BoxData> > pred_boxes;
    //GetPredBox(side_, num_object_, num_class_, input_data + input_index, &pred_boxes, sqrt_, constriant_, score_type_, nms_);
    GetPredBox(side_, num_object_, num_class_, swap_data + input_index, &pred_boxes, score_type_, nms_, biases_);

    int index = top_index + num_class_;
    int pred_count(0);
    for (std::map<int, vector<BoxData> >::iterator it = pred_boxes.begin(); it != pred_boxes.end(); ++it) {
      int label = it->first;
      vector<BoxData>& p_boxes = it->second;
      if (gt_boxes.find(label) == gt_boxes.end()) {
        for (int b = 0; b < p_boxes.size(); ++b) {
          top_data[index + pred_count * 4 + 0] = p_boxes[b].label_;
          top_data[index + pred_count * 4 + 1] = p_boxes[b].score_;
          top_data[index + pred_count * 4 + 2] = 0; //tp
          top_data[index + pred_count * 4 + 3] = 1; //fp
          ++pred_count;
        }
        continue;
      } 
      vector<BoxData>& g_boxes = gt_boxes[label];
      vector<bool> records(g_boxes.size(), false);
      for (int k = 0; k < p_boxes.size(); ++k) {
        top_data[index + pred_count * 4 + 0] = p_boxes[k].label_;
        top_data[index + pred_count * 4 + 1] = p_boxes[k].score_;
        float max_iou(-1);
        int idx(-1);
        for (int g = 0; g < g_boxes.size(); ++g) {
          float iou = Calc_iou(p_boxes[k].box_, g_boxes[g].box_);
          if (iou > max_iou) {
            max_iou = iou;
            idx = g;
          }
        }
        if (max_iou >= threshold_) { 
            if (!records[idx]) {
              records[idx] = true;
              top_data[index + pred_count * 4 + 2] = 1;
              top_data[index + pred_count * 4 + 3] = 0;
            } else {
              top_data[index + pred_count * 4 + 2] = 0;
              top_data[index + pred_count * 4 + 3] = 1;
            }
        }
        ++pred_count;
      }
    }
  }
}

INSTANTIATE_CLASS(EvalDetectionLayer);
REGISTER_LAYER_CLASS(EvalDetection);

}  // namespace caffe
