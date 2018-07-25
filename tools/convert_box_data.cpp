//  图片 + 边框标签 -> Datum datum: 
// 图像像素数据域  datum->set_data();
//  datum->add_float_data()  label + difficult + box[4]
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include <unistd.h>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, true,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "jpg",
    "Optional: What type should we encode the image as ('png','jpg',...).");
//********************************* label_file **************************************//
DEFINE_string(label_file, "",
    "a map from name to label");
//********************************************************************************//

int main(int argc, char** argv) {
#ifdef USE_OPENCV
 // glog 日志等级
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_box_data [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  // 解析命令行参数
 gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;// 彩色图 3通道
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;//
  const string encode_type = FLAGS_encode_type;
  //*************************************exit label_file*****************************************//
  const std::string label_file = FLAGS_label_file;
  if (label_file == "") {
    LOG(ERROR) << "empty label file";
    return 1;
  }
  //********************************************************************************//

  //******************************** produce label_file map***************************************//
  std::ifstream labelfile(label_file.c_str());
  std::map<std::string, int> label_map;
  std::string tmp_line;
// 标注文件 name标签 : class_id
/*    前面一列是标注文件内的类别string   dog:2  cat:6
      后面一列是实际 类id
1 1
2 2
3 3
4 4
5 5
6 6
7 7
8 8
9 9
10 10
*/
  while (std::getline(labelfile, tmp_line)) 
  {
    size_t pos = tmp_line.find_last_of(' ');
	// 标注文件 name标签 : class_id
    label_map[tmp_line.substr(0, pos)] = atoi(tmp_line.substr(pos+1).c_str()); 
  }
  //********************************************************************************//

  //************************************pair(jpg,xml)**************************************//
  std::ifstream infile(argv[2]);
  // 图片路径 和 对于 xml格式标签文件路径 对
  std::vector<std::pair<std::string, std::string> > lines;
  std::string line;
  size_t pos;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    std::string jpg = line.substr(0, pos);
    std::string xml = line.substr(pos+1);
    if((access(jpg.c_str(),0)==-1) || (access(xml.c_str(),0)==-1) ) continue;
    lines.push_back(std::make_pair(jpg, xml));
  }
  
  //********************************************************************************//
  // 打乱数据
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

// 尺寸变形 固定到网络输入大小
  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB 创建数据库文件
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db 保存
  std::string root_folder(argv[1]);
 
// 主要区别============== 
  // AnnotatedDatum anno_datum;
  // Datum* datum = anno_datum.mutable_datum();
  Datum datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;
  
// 遍历每一张数据===========================================
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
	// 图片编码格式==========================
    if (encoded && !enc.size()) 
	{
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;// 图片
      size_t p = fn.rfind('.');// 图片格式
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    //读取 图片+标签数据 转换成datum   **********************************************//
    status = ReadBoxDataToDatum(
	    root_folder + lines[line_id].first, // 图片路径
        root_folder + lines[line_id].second,// xml标注文件路径
		label_map,                          // 标注文件 name标签 : class_id
        resize_height,                      // 尺寸变形 固定到网络输入大小
		resize_width, 
		is_color,                           // 彩色图像
		enc,                                // 编码格式
		&datum);                            // 转换到的数据
		
// opencv 读取图像并变形 
// 图像编码 cv::imencode()
// 设置datum的data图像像素数据域  datum->set_data();
// 读取 xml 格式 的 标签数据
// object 域标注框数据 object.name 映射 name:id 获取真实 类别id
// object.bndbox 获取标注框(左下角、右上角坐标)
// 转换成 中心点坐标 和 变成尺寸 用图像尺寸归一化到0~1之间
// 设置label  标签 datum->add_float_data(float(label));
// 难度?  datum->add_float_data(float(difficult));  
// 边框信息 datum->add_float_data(box[i]);	//box[4]  x,y,w,h  (范围0~1)

// io.cpp ： ReadBoxDataToDatum()
    //************************************************************************************//
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) 
	  {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } 
	  else 
	  {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential 每一行id_图片路径
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;
//////////////////////// 存入LMDB 数据库文件中==============
    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) 
	{/////////////// 每转换 1000张数据 显示一次信息  并保存数据
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch  最后一些图片数据 不一定有1000张
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
