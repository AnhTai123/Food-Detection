# Food-Detection using Yolov8

## Initial Steps
To kickstart the process of food detection using Yolov8, follow these initial steps:

1. **Gắn ổ đĩa vào sổ ghi chép Colab**:
Đảm bảo bạn đã gắn ổ đĩa vào môi trường sổ ghi chép Colab để truy cập các tệp và thư mục cần thiết.
2. **Cài đặt Mô hình Yolov8**:
Cài đặt mô hình Yolov8 vào thư mục đích của Google Drive, nơi tập dữ liệu được tải. Bước này rất quan trọng cho việc huấn luyện và suy luận sau này.
3. **Cấu hình Tham số Huấn luyện**:
Trong hàm huấn luyện của mô hình Yolov8, hãy thêm đường dẫn data.yaml. Tùy chỉnh các tham số liên quan như epoch, cài đặt được huấn luyện trước, độ kiên nhẫn, v.v. để phù hợp với yêu cầu cụ thể của bạn.

## Dataset
Chuẩn bị và sắp xếp tập dữ liệu của bạn theo các hướng dẫn sau:

1. **Tải xuống Tập dữ liệu**:
Tải xuống tập dữ liệu theo định dạng được cung cấp. Đảm bảo tập dữ liệu có thể truy cập và lưu trữ phù hợp.
2. **Tải Tập dữ liệu lên Google Drive**:
Thêm tập dữ liệu vào Google Drive của bạn.

### Cấu trúc của tập dữ liệu
Sắp xếp tập dữ liệu của bạn theo cấu trúc thư mục sau:
```bash
 |__dataset
 |   |__images
 |   |   |__train
 |   |   |__test
 |   |   |__valid 
 |   |
 |   |__labels
 |       |__train
 |       |__test
 |       |__valid 
```

* Tập dữ liệu xác thực là tùy chọn, nhưng thư mục huấn luyện và kiểm tra là bắt buộc để huấn luyện mô hình.

* Cấu trúc bao gồm hai thư mục chính: 'images' và 'labels'.

Mỗi thư mục chứa các thư mục con để huấn luyện, kiểm tra và tùy chọn là dữ liệu xác thực.

Đảm bảo tập dữ liệu của bạn tuân thủ cấu trúc này để huấn luyện liền mạch với mô hình Yolov8.

* Lưu ý: Mặc dù tập dữ liệu xác thực là tùy chọn, nhưng chúng tôi khuyến nghị sử dụng để nâng cao hiệu suất mô hình.

Tuy nhiên, thư mục 'train' và 'test' là bắt buộc để huấn luyện mô hình thành công.

* Nếu tập dữ liệu được cung cấp quá lớn để tải xuống trực tiếp, chúng tôi có thể cung cấp tệp zip chứa dữ liệu cần thiết.

Đảm bảo cấu trúc tập dữ liệu được duy trì trong tệp zip để xử lý trơn tru.
## Resources
1. [Official Doucumentation Ultralytics](https://github.com/ultralytics/ultralytics)
2. https://docs.ultralytics.com/models/yolov8/#__tabbed_2_1
3. [Dataset](https://universe.roboflow.com/ahmad-nabil/food-detection-for-yolo-training)

## License
This project is licensed under the [MIT License](LICENSE).
