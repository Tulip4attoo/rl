# rl
My personal note on learning reinforcement learning


## Motivation

RL là thứ mà mình đã mong muốn học từ lâu, và thực tế là đã có 2 lần học (tháng 11-2017 và tháng 4-2018), tuy nhiên cả 2 lần mình đều bỏ giữa chừng (dù đã giành tương đối thời gian).

~~Lần thứ 3 này với mục tiêu không tiến nhanh nhưng tiến đều, kỳ vọng sẽ nắm được cơ bản của RL để có thể ứng dụng sau này.~~

Lần này mình sẽ dive vào để co thể sẵn sàng làm bot cho mọi game mình hướng tới. Như vậy, sẽ phải làm quen với nhiều oại môi trường (focus unity) và điều chỉnh được reward theo ý mình.

## Learning Plan

Mình sẽ học theo 2 cái chính:

+ [Khóa CS 294 của Berkerley](http://rail.eecs.berkeley.edu/deeprlcourse/): về cơ bản thì mình nghĩ học khóa nào cũng có ích thôi. Và cơ bản khóa này đang diễn ra, tạo cảm giác học đuổi sẽ thích hơn. Khóa có 28 lectures, 5 homeworks + 1 project.

~~+ nhóm học ở công ty~~

+ làm theo các tutorial ở [A Free course in Deep Reinforcement Learning from beginner to expert.
](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)

+ tự làm các prj theo các tookit khác nhau (gym, MAME,...)

Tạm sẽ gắng follow theo khóa CS 294 - 1 tuần 2-3 videos, nghĩa là tốc độ bám sát hoặc nhanh hơn tốc độ gốc 1 chút. Sẽ cập nhật tiến độ học ở file readme này, về bài tập, tóm tắt nội dung bài giảng, mình sẽ tạo các folder/file để cập nhật.

## Tracking plan

+ CS 294: xong lesson 1
+ các project: 
    
    + [taxi-v2](https://github.com/Tulip4attoo/rl/tree/master/f-class/taxi-v2) (done)

    + [cartpole](https://github.com/Tulip4attoo/rl/tree/master/f-class/cartpole) (done)

    + pong-v0 (doing)

    + street fighter (to do)
+ các bài viết:

    + [bài giới thiệu về OpenAI.](https://tulip4attoo.github.io/blog/lam-quen-openai-gym/)
    + [tóm tắt TIR về tình hình ứng dụng RL hiện tại](https://tulip4attoo.github.io/blog/tir-rl-hard/)
    + [quản lý memory](https://tulip4attoo.github.io/blog/til-manage-memory-in-rl/)

## Some goals

- implement tetris RL at the end of Nov.
- implement some bot of some games at the end of Dec.

## Timeline

- 24 Oct 18: init
- 28 Oct 18: học xong lesson 1 CS294
- 30 Oct 18: kick off nhóm RL ở công ty. Viết xong [bài giới thiệu về OpenAI.](https://tulip4attoo.github.io/blog/lam-quen-openai-gym/)
- 31 Oct 18: code xong bài [taxi-v2](https://github.com/Tulip4attoo/rl/tree/master/f-class/taxi-v2), sử dụng q learning. Remind lại khái niệm q-table. Lần đầu áp dụng thực hành 1 bài RL.
- 31 Oct 18: code xong bài [cartpole](https://github.com/Tulip4attoo/rl/tree/master/f-class/cartpole). Lần đầu tiếp xúc với khái niệm DQN (deep q network). Dùng code từ 1 bài có architect khác, đổi architect từ có dùng CNN và input image sang chỉ dùng dense cho input size (1,4). Sửa code mệt nghỉ.
- 01 Nov 18: thuyết tình về 2 projects với team nhưng hơi fail. Chú ý tới vấn đề tại sao weight 1 đằng code 1 nẻo (về w_{i-1} / w_{i}). Tuy nhiên chỉ dừng lại ở chú ý chứ chưa biết làm như thế nào =))
- 02 Nov 18: đọc bài và hiểu thêm chút chút về DRL, cũng như hiểu được tầm đột phá của RL khi có thể dùng được transfer learning (giờ thì chưa)
- 02 Nov 18: lựa chọn [Pong-v0](https://gym.openai.com/envs/Pong-v0/) làm project tiếp theo (sử dụng images làm input). Ngoài ra biết thêm về 1 hệ môi trường mới ([MAMETookit](https://github.com/M-J-Murray/MAMEToolkit)), có thể chơi được game arcade. 
- 04 Nov 18: họp nhóm RL thực hành ở công ty. Cũng ko ổn lắm, mà kệ. Viết bài [tóm tắt TIR về tình hình ứng dụng RL hiện tại](https://tulip4attoo.github.io/blog/tir-rl-hard/). Nói chung mấy hôm rồi chưa học được mấy.
- 05 Nov 18: học xong l2 của CS294. Do quá bận nên chưa thể làm file note nội dung ra được... Không biết sau có thể làm không?
- 07 Nov 18: học xong l3 và 1 nửa l4. 
- 08 Nov 18: nhóm học trên công ty, nhưng không hiệu quả mấy. Buổi này nhóm kia nói về bài cartpole nhưng không có gì đáng chú ý. Nghiêm túc suy nghĩ về việc nên học nhóm thế nào?
- 11 Nov 18: dự định viết bài hướng dẫn taxi nhưng chưa hoàn thành.
- 14 Nov 18: trao đổi với Thịnh, nhận thấy thực ra cuốn RL an intro có vẻ cũng ổn chứ không phải ko, tuy rằng nó không focus vào kxy thuật mới lắm, nhưng thực ra các kỹ thuật mới build trên nền các kỹ thuật cơ bản từng có thôi. Hmmm
- 15 Nov 18: trao đổi với anh Trung, nhận thêm được động lực cũng như phương hướng làm. Hôm nay nhóm học trên công ty hoãn. Mình đang tính tới việc không học cùng mọi người nữa (pace khác quá, mà nếu bảo mình guide thì cũng khó vì như thế tốc độ sẽ chậm đi nhiều. Hmmm). Tối thì đã code xong Pong để train.
- 16 Nov 18: train Pong ok, sau 200 episode thì tạm ổn, có triển vọng sẽ win được game ở 1000 episode gì đó, có thể sẽ bỏ lên colab/kaggle kernel để train??? Đồng thời nhận thấy vấn đề [quản lý memory](https://tulip4attoo.github.io/blog/til-manage-memory-in-rl/). 
- 16 Nov 18: Code cách 2 mà chưa xong, đau đầu dã man. Nghiêm túc nghĩ tới việc refactor và chuyển về OOP toàn bộ code, nhưng chắc để xong cách 2 thì mới làm. Nếu mà mình xong, thì sẽ có cả 1 framework/lib để train cho mấy bài dạng này luôn.
- 16 Nov 18: nhận thấy có vẻ hàm reward của game Pong có vấn đề, tuy nhiên không lục ra được chỗ nào có document cụ thể về hàm reward cả. Hmmm. Ngoài ra suy nghĩ tới việc làm cách nào để thay đổi được hàm reward, bởi lẽ nếu hàm reward kia không hợp lý thì sao? Đó là 1 vấn đề dài đáng suy nghĩ.
- 16 Nov 18: thử lập acc azura mà không được. Rất cần 1 thẻ VISA để có thể lập các acc GCP/AWS/Azure để train thử. Hmmm. Tạm tính kế train trên colab/kaggle kernel
- 18 Nov 18: đã code xong phần memory lưu ra file. Thiết kế riêng class memory, bỏ vào 1 file riêng để có thể sử dụng tiện hơn cho những lần sau. Nếu thiết kế được kiểu config nữa thì quá tuyển.
- 19 Nov 18: phat hiện ra kiểu save file và load chạy cực kỳ cực kỳ chậm. Suy nghĩ tới việc chuyển load cho 10/100 batch cho từng lần chạy, chú ý 100 batch thực ra chỉ là random trong 100/1mil = 0.0001 nên ảnh hưởng hầu như không đáng kể, có thể thực hiện được. Tuy nhiên, mình đọc được bài này: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb Ở đây có 1 cách khá hay là chuyển về dạng uint8 (so với np.float16 của mình), cũng như chuyển dạng lưu là 1mil frame, và khi load chỉ cần index sẽ kéo ra 5 frame, tạo thành state và next_state. Như vậy, memory_size có thể tăng lên thêm 2 * 8 = 16 lần, như giờ mình có thể train với 35k memory_size --> 35k*16 = 560k, có thể coi là đã ổn rồi. Cheer! Vậy là mình nên đọc qua thêm vài github để coi họ có các hướng nào hay ho để học tập nữa, hehee.
- 10 Dec 18: do chuyện cá nhân nên bỏ bê hơi lâu, code được phase 1 và phase 2 của cfg. Nếu phase 3 ok nữa thì sẽ triển luôn vào những dự án tiếp.
- 18 Dec 18: nghĩ về chuyện chuyển qua code bằng pytorch