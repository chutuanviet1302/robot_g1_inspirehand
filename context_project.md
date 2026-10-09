# HomeHand — context & trạng thái dự án

Cập nhật: 2026-10-09 (phiên 4). Thư mục: `/home/tuanviet2003/Downloads/Project_Robotics`.
Máy: i5-11400H (6C/12T), GTX 1650 4 GB, RAM 16 GB, Ubuntu (đích: chạy được cả Windows).

## 1. Dự án là gì

Robot **Unitree G1** (đứng cố định, 29 DoF nhưng chỉ 2×7 khớp tay hoạt động) gắn **2 bàn tay Inspire** (mỗi tay 6 actuator),
trong **MuJoCo**. Nhiệm vụ: nhìn bàn bếp bằng camera đầu, **nhận diện** các vật YCB (lon súp, chai mustard, hộp đường, hộp thịt),
**vật nào gần tay nào thì tay đó nắm**, hai tay luân phiên, **thả vật vào thùng**, dọn hết bàn.
Có web app (FastAPI + React) để xem trực tiếp, đánh giá hàng loạt, xem lại episode.
Mục đích: dự án portfolio xin việc robotics, không có robot thật (đánh giá "sim gap" bằng domain randomization).

## 2. Kiến trúc

```
camera đầu RGB-D → YOLOv8n-seg → mask + depth → ước lượng 3-D (vị trí, chiều cao, yaw)
   → planner (tay gần hơn, vật ngoài rìa trước, nhìn lại sau mỗi lần gắp)
   → skill gắp-thả: expert (IK) | ACT | Diffusion Policy   (13 mục tiêu khớp / tay)
   → safety layer (giới hạn khớp, tốc độ, va chạm, lực, workspace, dừng bảo vệ)
   → MuJoCo (26 actuator vị trí, vật lý 500 Hz, điều khiển 25 Hz)
```

## 3. Cấu trúc code (`homehand/`, ~4300 dòng Python, 40 file)

| Thư mục / file | Việc |
|---|---|
| `model/build_repo.py` | **Dựng robot từ model G1+Inspire do bạn tải** (`third_party/unitree_g1_inspire`): hàn chân/eo, đổi motor → position actuator, đổi tên khớp, sửa mimic bị đảo, thêm site lòng bàn tay/đầu ngón |
| `model/build.py` | Dựng cảnh bếp + vật YCB; builder dự phòng (Menagerie G1 + tay dex-urdf, chạy được sau `fetch-assets`) |
| `model/objects.py`, `control/grasps.py` | 4 vật, mỗi vật một kiểu nắm: cylindrical wrap, low power wrap, palmar box, tripod pinch |
| `control/ik.py` | IK mink 2 tay, tư thế chờ tự nhiên, phạt vặn cổ tay, né va chạm tay–thân (cách 6 cm) |
| `control/skills.py`, `planner.py` | Skill gắp-thả (state machine) và bộ lập kế hoạch dọn bàn |
| `control/safety.py` | Lớp an toàn giữa mọi controller và robot |
| `env/kitchen_env.py`, `randomize.py` | Môi trường, phân loại kết quả, 4 mức randomization (nominal/low/medium/high) |
| `perception/` | Camera, detector (YOLO + oracle), định vị 3-D, sinh dữ liệu synthetic + train YOLO |
| `policy/` | ACT, Diffusion Policy, train chung, chạy policy làm skill |
| `data/record.py` | Thu demo từ expert (chỉ giữ lần gắp thành công) |
| `eval/` | Wilson CI, SQLite, chạy song song có giới hạn RAM |
| `server/`, `web/` | API + WebSocket stream; React 4 tab (Live, Evaluation, Episodes, Models), build sẵn vào `homehand/web_static` |
| `cli.py` | `homehand fetch-assets / build-model / view / demo / collect / train / eval / perception ... / pipeline / serve` |
| `tests/test_core.py` | 10 test (đang pass) |
| `.github/workflows/ci.yml` | CI Ubuntu + Windows |
| `scripts/report.py` | Sinh `RESULTS.md` từ database |

## 4. Đã làm xong và kiểm chứng

- Model G1 + 2 tay Inspire biên dịch ổn, 26 actuator, 12 ràng buộc mimic; mô phỏng ổn định.
- **Lỗi trong model của repo bạn tải:** ràng buộc mimic bị đảo chiều (ngón gập thiếu 25–45%). Đã đảo lại đúng URDF.
- Hướng gắn tay (ngón ra trước, ngón cái lên, lòng bàn tay vào trong) trùng với cách gắn của mình (đối chiếu ma trận xoay).
  Khe hở cổ tay–bàn tay 8 mm đã được đóng lại (đặt tay lún 4 mm); **chưa render lại cận cảnh để xác nhận bằng mắt**.
- Ngón tay xuyên vật: đo hình học cho cả 4 vật, độ lún ngón ≤ 0.4–1 mm (chỉ cổ tay/cẳng tay có chỗ lún ≤ 3 mm).
  Chưa tái hiện được hiện tượng bạn thấy trong ảnh; đã giảm lực kẹp xuống ~13 N ở đầu ngón (tay thật ≤ ~10 N).
- Tư thế tay: tư thế chờ định nghĩa theo góc khớp tự nhiên (cổ tay thẳng 0°); trước đó cổ tay vặn ~70°, khuỷu dạng ra 15 cm.
- RAM: mỗi worker ~0.8 GB (trước 1.7 GB); số worker tự giới hạn theo RAM trống (`resources.py`).
- Perception với mask oracle: sai số vị trí trung bình khoảng 2–13 mm (đo trước khi đổi bố trí vật); YOLO **chưa train**.
- Web app chạy, trình duyệt headless tải trang không lỗi console (kiểm tra trước khi đổi model sang repo của bạn).
- Torch 2.5.1+cu121 nhận GPU; ultralytics và torchvision đã cài.


## 4b. Phiên 2026-10-08: chuyển động tự nhiên, chống xuyên, Docker

- **Quỹ đạo mới** (`control/skills.py`, `control/ik.py: plan_path`): mỗi pha là một chuyển động liền, đi qua
  điểm trung gian bằng góc bo (không dừng), lấy mẫu 2 cm + IK hội tụ đầy đủ *trước* khi chạy (không IK lúc chạy),
  thời gian **minimum-jerk** theo max(tốc độ tay 0.35 m/s, tốc độ khớp 90% giới hạn, tốc độ xoay cổ tay).
  Nguyên nhân giật cũ: nội suy tuyến tính dừng-chạy + IK kéo posture làm vai-xoay/cổ tay-lăn xoắn ngược nhau 10 rad/s.
- **Chọn góc xoay tay theo tầm với thật** (`model/objects.py: hand_yaw_window`): tay phải ở |y|=0.27 chỉ xoay
  [-35°, 0°], ở |y|=0.13 [-15°, +25°] (đo bằng IK). Trước đó mustard ô ngoài không với tới (lệch 5–8 cm) → làm đổ chai.
- REACH_Z 1.13 → 1.03 (điểm trung gian cũ ngang vai sát ngực, IK không với tới); thả cao 4.5 cm trên mép rổ, rút thẳng lên.
- **Chống xuyên**: lòng bàn tay va chạm bằng convex hull của mesh hiển thị (trước: mesh hiển thị lún 5 mm); planner bỏ qua
  vật nằm sát rổ nếu đầu ngón sẽ chạm thành rổ (`planner.clear_of_bin`). Lún tối đa đo trên 24 episode: ~1 mm.
- Horizon: tidy 400 bước/vật, pick_place 360.
- Số liệu 24 seed (expert, ground-truth, nominal): **13/24 sạch (54%), 81–84/96 vật (84–88%), 0 dừng an toàn**,
  gia tốc khớp RMS 2.7 rad/s² (cũ 9.5). Lỗi còn lại chủ yếu hộp thịt (tripod pinch) rơi/đặt lệch.
- **Docker**: `Dockerfile`, `docker-compose.yml` (web, view qua X11, test), `.dockerignore`; assets mount từ host
  (không đóng model không giấy phép vào image). Build dùng `network: host` vì mạng bridge làm đứt tải wheel lớn.
- **Dataset demo cũ phải thu lại** (expert đã đổi) trước khi train ACT/Diffusion.

## 4c. Phiên 2026-10-08 (tiếp): nắm trượt, pipeline, bug sim-gap

- `homehand grasp-bench` (mới, `eval/grasp_bench.py`): gắp từng vật trên lưới vị trí × góc, phân loại lỗi
  missed/pushed/slipped/bounced. Kẹp 3 ngón hộp thịt chỉnh lại (ngón cái 1.3 rad, áp út/út 1.3, nắm thấp 4.5 cm):
  63/72 → 69/72. Toàn bộ: 186/192 (97%).
- Planner kiểm tra va chạm rổ bằng động học thật (IK + vị trí đầu ngón), nếu chặn thì thử cổ tay không xoay vào trong.
- `scripts/motion_quality.py`: 24 episode expert (ground truth): **21/24 sạch, 93/96 vật, 0 dừng an toàn**,
  gia tốc khớp RMS 2.7 rad/s², lún tối đa 0.9 mm. Kết quả lưu `data/motion_quality.json`.
- Pipeline (300 episode thu demo = 846 demo/268k khung; YOLO mAP50 0.995; ACT 10k bước 24 phút;
  Diffusion 10k bước 39 phút; GPU không lỗi Xid khi tắt AMP).
- Đánh giá (YOLO, nominal, 20 episode): expert 90% sạch / 97% vật; ACT 70% / 90%; Diffusion 0% / 1% (10 bước DDIM)
  → 0% / 30% (30 bước). Expert + oracle 80% / 95%. Diffusion v2 (train tiếp tới 30k bước, 30 bước DDIM): **45% sạch / 78% vật, 0 dừng an toàn**.
- **BUG nghiêm trọng đã sửa**: randomization khối lượng đổi `body_mass` mà không gọi `mj_setConst` → subtreemass /
  invweight0 sai → +20% khối lượng đã làm vật trượt và mô phỏng NaN. Mọi kết quả sim-gap trước đó (kể cả số trong
  context cũ) đo nhầm bug. Đã sửa + test hồi quy; các run sim-gap cũ đánh dấu `invalid_mass_bug` trong DB.
  Dataset expert_v1 thu ở mức "low" có bug này (chỉ giữ demo thành công nên vẫn dùng được).
- RAM: worker có PyTorch/YOLO chiếm 2.1 GB (không phải 0.9) → `WORKER_TORCH_MB` trong `resources.py`.
- Video: `media/demo_seed5.mp4` (YOLO, 4/4). Cài thêm `imageio` + `imageio-ffmpeg`.
- Docker: wheel tải trước bằng `scripts/docker_wheels.sh`; render headless OSMesa; test/web chạy được trong container.
- Còn lại / ý tưởng: tư thế chờ (tay dang ngang ngực) chưa tự nhiên — đổi thì phải thu + train lại;
  VLA (SmolVLA) chưa làm; chưa test trên Windows thật.

## 4d. Phiên 2026-10-08 tối / 10-09: theo nhận xét của bạn sau khi xem viewer

Nhận xét: (1) thả cùng một chỗ, (2) tư thế khớp không tự nhiên, (3) chỉ một kiểu nắm ngang, vật nằm phải nắm khác.
Tham khảo repo `chutuanviet1302/openarm-rh56-pick-place` (thư viện nắm theo tư thế nằm/đứng, ngàm đo bằng FK,
offset vật trong tay đo thật để tính điểm thả).

- **Thả nhiều chỗ** (bạn yêu cầu đơn giản): 3 điểm/tay ngang rổ, cách thành gần 3.5 cm, chọn điểm xa vật đã có trong rổ
  (`planner.choose_drop_spot`). Rổ dời 0.62 → **0.56 m** (lòng bàn tay G1 chỉ với tới x ≈ 0.46 m ở độ cao thả);
  ô trong dời x 0.34–0.40 → **0.30–0.35**.
- **Tư thế tự nhiên**: `ArmIK.NATURAL` = cánh tay trên buông dọc thân, khuỷu ở hông, cẳng tay chếch lên 45°, cổ tay thẳng,
  ngón hơi cong (`RELAXED_HAND`); tay chờ luôn cao hơn mọi vật. Tách `ArmIK.WORK` (tư thế tham chiếu IK khi làm việc).
  Với tay / mang / quay về đi trong **không gian khớp** qua điểm cao (1.12–1.22 m); tiếp cận / hạ / nhấc theo đường thẳng.
  Bài học: tư thế chờ thấp (z 1.03) làm tay quẹt hộp đường; nội suy khớp cả đoạn hạ làm tay đi chéo đè vật.
- **Vật nằm + nắm từ trên**: `objects.LYING` (tư thế nằm từng vật), `rest_of`, `body_pose_for`; env sinh ~50% vật ở ô ngoài
  nằm (x 0.33–0.37; vật dài nằm theo trục y; lon/hộp thịt hướng trong ±40° vùng với tới). `grasps.LYING_GRASP`: 4 kiểu nắm
  từ trên (tham số tìm bằng `scripts/top_grasp_search.py`). Mấu chốt: **ngón cái xoay vào trước khi hạ tay** (nếu không
  ngón cái bị mặt bàn chặn, các ngón kéo vật tuột); tay trái đối xứng theo **pháp tuyến lòng bàn tay (-y)**, không phải trục ngón
  cái; điểm thả tính từ **vị trí vật thật trong tay** (`held_in_palm`); khi mang, cổ tay lật gần phẳng để với tới rổ.
- **Perception**: `localize` nhận đứng/nằm từ độ cao đo được và khớp hình chữ nhật cho vật nằm; YOLO train lại với ảnh có vật
  nằm: mAP50 0.995, nhận đúng đứng/nằm 97.9%, sai số vị trí 3.6 mm.
- **Đặc trưng policy** đổi (39 chiều): thêm cờ vật nằm, tư thế nắm + điểm thả theo kế hoạch (tính từ perception), one-hot 8
  kiểu nắm → phải thu lại demo (đã thu: 862 demo / 314k khung) và train lại ACT / Diffusion (30k bước).
- **Số liệu expert mới** (ground truth): grasp-bench **188/192** (vật nằm 63/64); 24 episode dọn bàn **18/24 sạch, 88/96 vật,
  0 dừng an toàn**, gia tốc khớp RMS 2.0 rad/s², lún tối đa 2 mm (cổ tay chạm hộp đường đang cầm).
  Sim-gap tách yếu tố: ma sát ×0.5 hại nhất (27/32).
- Lỗi của tôi trong phiên: `grasps.py` lỗi cú pháp làm pool worker khởi động lại liên tục (treo 20 phút, tải 17) → nhớ
  `python -c "import homehand.control.grasps"` trước khi chạy song song.
- Viewer giờ in lý do dừng an toàn. 2 lần dừng an toàn thấy trong viewer (seed 23, 27) không tái hiện được khi chạy nền.

## 5. Con số hiện tại (CŨ, trước phiên 2 — sim-gap trong đó bị bug khối lượng) (expert, oracle perception, nominal)

| Phép đo | Kết quả | Ghi chú |
|---|---|---|
| Gắp-thả 1 vật, vị trí gần giữa | 30/32 (94%) | model repo bạn, trước khi đổi bố trí |
| Gắp-thả 1 vật, vị trí ngoài rìa (|y| 0.25–0.29) | 25/32 (78%) | |
| Dọn cả bàn, 4 vật (bố trí hiện tại) | **6/30 episode sạch (20%)**, **74/120 vật vào thùng (62%)** | 0 lần dừng an toàn; lỗi: grasp_fail 7, dropped 8, misplaced 4, knocked_over 5 |
| Dọn cả bàn, 3–4 vật đặt gần thân hơn (bố trí cũ) | 14/30 (47%) và 76% vật | bạn yêu cầu dời vật ra xa nên bỏ bố trí này |

Hộp thịt (nắm 3 ngón) yếu nhất, đặc biệt tay trái. **Chưa có số liệu nào của ACT, Diffusion Policy hay YOLO.**

## 6. Trạng thái pipeline (2026-10-09, 11:30)

Kết quả lần chạy trước (expert cũ, chưa có vật nằm) ở `data/archive/run1_2026-10-08/`. Lần này:
- Đã xong: thu demo (862 demo), ảnh YOLO, YOLO (mAP50 0.995, đứng/nằm đúng 97.9%), benchmark, ACT (10k bước, val 0.0076).
- **Máy khởi động lại 2 lần** (09:52 và ~11:10, đều là reboot có trật tự, không phải lỗi GPU) → mất tiến độ Diffusion.
  Đã thêm **checkpoint huấn luyện** (`models/<tên>/training_state.pt` mỗi lần log, tự resume nếu cùng cấu hình).
- **Lỗi policy đã sửa:** `PolicySkill` chạy cố định 532 bước/lần gắp → 4 vật vượt horizon. Giờ kết thúc khi tay về tư thế chờ
  (≥ 60% độ dài demo trung bình). ACT sau sửa (YOLO, nominal 20 ep.): 15% sạch / 59% vật / 3 dừng an toàn (lực > 80 N ấn
  bàn/rổ; lớp an toàn chặn đúng). Kém xa expert vì nhiệm vụ khó hơn (8 kiểu nắm, vật nằm) với cùng lượng dữ liệu.
- Đang chạy (scripts trong `data/logs/`): `pipeline.sh` (Diffusion 30k bước, có checkpoint) → `act_v2.sh` (ACT +15k bước
  từ v1 rồi đánh giá); `eval_act_rest.sh` (ACT medium/high); `eval_final.sh` (expert, Diffusion, expert+oracle, expert
  sim-gap, rồi `scripts/report.py`). Nếu máy khởi động lại: chạy lại đúng các script này (đều bỏ qua phần đã xong / resume),
  đánh dấu run DB còn `running` thành `aborted`.

## 7. Việc còn lại

1. Diffusion Policy: xem kết quả sau 30k bước; nếu vẫn kém ACT nhiều thì ghi rõ trong báo cáo.
2. Sim-gap: tăng số episode mỗi mức (hiện 10). Yếu tố quan trọng nhất: ma sát ngón–vật.
3. VLA (SmolVLA) chưa làm: cần ảnh + câu lệnh, fine-tune trên Kaggle.
4. Chưa chạy trên Windows thật; viewer trong Docker (X11) chưa thử.
5. Vật nằm chỉ ở ô ngoài và trong vùng hướng tay với tới (giới hạn động học G1 + tay úp) — ghi rõ khi trình bày.

## 8. Điều cần biết / lưu ý

- **Giấy phép:** model `unitree_g1_inspire` không có giấy phép; để trong `third_party/` (đã gitignore), không đưa vào repo công khai.
  Tay Inspire từ dex-urdf là CC BY-NC-SA (chỉ phi thương mại); YCB là CC BY 4.0.
- **Yêu cầu của bạn:** không tràn RAM, giới hạn an toàn khớp, có damping, an toàn robot (đã làm, xem `control/safety.py`, `resources.py`).
  Bạn không thích chờ lâu mà không có cập nhật: nên báo tiến độ ngắn thường xuyên.
- **Mẹo vận hành:** `pkill -f` có thể giết chính shell đang chạy lệnh; dùng `pgrep -f "tên[x]"` rồi kill theo PID.
  Worker sinh ra từ multiprocessing có thể mồ côi, kiểm tra bằng `ps` trước khi chạy tiếp.
  Viewer và `homehand serve` (1.4 GB) cùng chiếm RAM khi pipeline đang chạy.
- Server web (`homehand serve`) có thể vẫn đang chạy ở cổng 8000 với model cũ; khởi động lại sau khi pipeline xong.
- Repo tham khảo: `feraco/unitree_g1_inspire` (nguồn model đang dùng), `luckyrobots/g1-manipulation-challenge`
  (tay Dex3, chỉ lấy tư thế mặc định), `unitreerobotics/dfx_inspire_service` (giao diện tay thật: 6 lệnh chuẩn hoá [0,1],
  lực tối đa ~9.8 N mỗi ngón; mình mới dùng làm căn cứ giảm lực kẹp, chưa viết lớp chuyển đổi lệnh sang tay thật).
