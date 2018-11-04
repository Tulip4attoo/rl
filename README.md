# rl
My personal note on learning reinforcement learning


## Motivation

RL là thứ mà mình đã mong muốn học từ lâu, và thực tế là đã có 2 lần học (tháng 11-2017 và tháng 4-2018), tuy nhiên cả 2 lần mình đều bỏ giữa chừng (dù đã giành tương đối thời gian).

Lần thứ 3 này với mục tiêu không tiến nhanh nhưng tiến đều, kỳ vọng sẽ nắm được cơ bản của RL để có thể ứng dụng sau này.

## Learning Plan

Mình sẽ học theo 2 cái chính:

+ [Khóa CS 294 của Berkerley](http://rail.eecs.berkeley.edu/deeprlcourse/): về cơ bản thì mình nghĩ học khóa nào cũng có ích thôi. Và cơ bản khóa này đang diễn ra, tạo cảm giác học đuổi sẽ thích hơn. Khóa có 28 lectures, 5 homeworks + 1 project.

+ nhóm học ở công ty

+ làm theo các tutorial ở [A Free course in Deep Reinforcement Learning from beginner to expert.
](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)

Tạm sẽ gắng follow theo khóa CS 294 - 1 tuần 2-3 videos, nghĩa là tốc độ bám sát hoặc nhanh hơn tốc độ gốc 1 chút. Sẽ cập nhật tiến độ học ở file readme này, về bài tập, tóm tắt nội dung bài giảng, mình sẽ tạo các folder/file để cập nhật.

## Tracking plan

+ CS 294: xong lesson 1
+ các project: 
    
    + [taxi-v2](https://github.com/Tulip4attoo/rl/tree/master/f-class/taxi-v2) (done)

    + [cartpole](https://github.com/Tulip4attoo/rl/tree/master/f-class/cartpole) (done)

    + pong-v0 (doing)

    + street fighter (to do)

## Some goals

- implement tetris RL at the end of Nov.
- implement some bot of some games at the end of Dec.

## Timeline

- 24 Oct 18: init
- 28 Oct 18: học xong lesson 1 CS294
- 30 Oct 18: kick off nhóm RL ở công ty. Viết xong [bài giới thiệu về OpenAI.](https://tulip4attoo.github.io/blog/lam-quen-openai-gym/)
- 31 Oct 18: code xong bài [taxi-v2](https://github.com/Tulip4attoo/rl/tree/master/f-class/taxi-v2), sử dụng q learning. Remind lại khái niệm q-table. Lần đầu áp dụng thực hành 1 bài RL.
- 31 Oct 18: code xong bài [cartpole](https://github.com/Tulip4attoo/rl/tree/master/f-class/cartpole). Lần đầu tiếp xúc với khái niệm DQN (deep q network). Dùng code từ 1 bài có architect khác, đổi architect từ có dùng CNN và input image sang chỉ dùng dense cho input size (1,4). Sửa code mệt nghỉ.
- 01 Nov 18: thuyết tình về 2 projects với team nhưng hơi fail. Chú ý tới vấn đề tại sao weight 1 đằng code 1 nẻo (về $w_{i-1} / w_{i}$). Tuy nhiên chỉ dừng lại ở chú ý chứ chưa biết làm như thế nào =))
- 02 Nov 18: đọc bài và hiểu thêm chút chút về DRL, cũng như hiểu được tầm đột phá của RL khi có thể dùng được transfer learning (giờ thì chưa)
- 02 Nov 18: lựa chọn [Pong-v0](https://gym.openai.com/envs/Pong-v0/) làm project tiếp theo (sử dụng images làm input). Ngoài ra biết thêm về 1 hệ môi trường mới ([MAMETookit](https://github.com/M-J-Murray/MAMEToolkit)), có thể chơi được game arcade. 