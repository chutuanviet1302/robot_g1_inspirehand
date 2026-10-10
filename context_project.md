# HomeHand — context & trạng thái dự án

Cập nhật: 2026-10-09 (phiên 5, Windows — xem §4e). Thư mục Linux: `/home/tuanviet2003/Downloads/Project_Robotics`.
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
| `data/record.py` (homehand/data) | Thu demo từ expert (chỉ giữ lần gắp thành công) |
| `eval/` | Wilson CI, SQLite, chạy song song có giới hạn RAM |
| `server/`, `web/` | API + WebSocket stream; React 4 tab (Live, Evaluation, Episodes, Models), build sẵn vào `homehand/web_static` |
| `cli.py` | `homehand fetch-assets / build-model / view / demo / collect / train / eval / perception ... / pipeline / serve` |
| `tests/test_core.py` | 16 test (pass trên Windows) |
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

## 4e. Phiên 2026-10-09 (phiên 5): chạy trên Windows 11 thật

Máy: Windows 11, Python 3.12 (`.venv`), MuJoCo 3.15, torch 2.5.1+cu121, ultralytics 8.4, GTX 1650. Thư mục
`D:\Program Files\Downloads\Vin Dynamics\g1_inspirehand\robot_g1_inspirehand`. `data/` và `models/` của máy Linux không
có ở đây → chạy lại toàn bộ bằng `scripts/run_all.py`.

- **Lỗi repo nghiêm trọng:** `homehand/data/record.py` không có trong git vì `.gitignore` dòng `data/` bỏ qua cả
  `homehand/data/` → `collect` / `train` / `pipeline` hỏng với bản đã push. Đã viết lại (shard theo seed để resume,
  chỉ giữ lần gắp cuối cùng `placed` + vật kết thúc trong rổ, `meta.json` đúng định dạng web/report đọc) và đổi
  `.gitignore` thành `/data/`, `/generated/`. Có test `test_dataset_merge_and_load`.
- Lỗi Windows đã sửa: scripts + test đặt `MUJOCO_GL=egl` vô điều kiện (MuJoCo Windows báo lỗi) → chỉ trên Linux;
  web app trang trắng vì Windows trả `.js` là `text/plain` (MIME từ registry) → `mimetypes.add_type` trong
  `server/app.py` + test hồi quy; `fetch-assets` treo vô hạn (`urlretrieve` không timeout) → `urlopen(timeout=60)`.
- Cài đặt Windows: torch wheel 2.4 GB bị khoá file (antivirus) → `pip download` rồi cài từ file; clone
  `feraco/unitree_g1_inspire` bị đứt HTTP/2 → `git -c http.version=HTTP/1.1` + partial clone, bỏ `media/`.
- **16/16 test pass trên Windows** (gồm render camera + test gắp đầy đủ). Viewer native chạy được: 28 episode
  expert xem trực tiếp, đa số 4/4, 0 dừng an toàn.
- Mới: `scripts/run_all.py` (pipeline + grasp-bench + motion quality + sim-gap ablation 16 seed/yếu tố + report,
  resume được); `control/inspire_real.py` (lệnh sim → giao diện RH56DFX: [0,1] 1 = mở, thứ tự út → xoay ngón cái,
  thanh ghi 0–1000, giới hạn lực ≤ 9.8 N, giới hạn tốc độ, giữ lệnh khi NaN, hook hiệu chuẩn; chưa thử tay thật).
- Pipeline lần này: 870 demo / 324k khung; YOLO mAP50 0.995, sai số 3.5 mm, đứng/nằm 97.9%; ACT 25k bước (trước
  10k), Diffusion 30k; sim-gap 25 ep/mức (trước 10).
- Bài học: viewer `--loop` chạy song song lúc train làm ACT chậm ~7× (tranh GPU + RAM) → không mở viewer khi train.

## 4f. Phiên 6 (2026-10-10): quỹ đạo giống người + SmolVLA (nhánh `human-motion`)

- **Chẩn đoán lỗi** (50 ep, ground truth, expert v1): 8/11 vật mất KHÔNG phải vật đang gắp — cẳng tay chúi xuống
  sau bàn tay lúc mang tới rổ quẹt đỉnh hộp đường ở ô trong (seed 0, 12, 24), ngón chạm vật bên cạnh khi tiếp cận.
- **Expert v2** (`control/human_motion.py`, `clearance.py`; `HOMEHAND_EXPERT_STYLE=v1` = bản cũ):
  ghép reach+descend và lift+transport thành 1 chuyển động; profile vận tốc bất đối xứng v~t²(1−t)³ (đỉnh 40 %);
  Fitts làm cận dưới thời lượng; preshape bàn tay trong lúc với tới; kiểm tra khoảng hở cổ tay/ngón với vật khác
  (nâng điểm thả, hoãn vật mà tư thế nắm chạm vật bên cạnh). Bài học: ghép luôn pha approach cuối làm tay PD
  cắt góc và đẩy vật cao (5/50 ep) → approach giữ riêng (`HOMEHAND_BLEND_APPROACH=0`).
- Số liệu: ground truth 50 ep: v1 185/200 → v2 188/200; YOLO 50 ep: v1 174/200 (66 % sạch) vs v2 174/200 (60 %,
  0 dừng an toàn) — ngang nhau. Giống người (`scripts/human_likeness.py`): mang vật 0 điểm dừng (v1: 1),
  SPARC −1.67 (92 % trong khoảng người; v1 −2.27, 0 %); với tới: đỉnh vận tốc ở 0.34 (người 0.35–0.45), còn 1
  điểm dừng có chủ đích trước khi tiến sát vật; tốc độ lòng bàn tay còn nhiều đỉnh do nội suy không gian khớp.
- **Render GPU**: trên laptop Optimus, MuJoCo render bằng Intel iGPU (250–380 ms/ảnh) → `SHIM_MCCOMPAT` trong
  `homehand/__init__.py` ép GTX 1650 (1.5–13 ms/ảnh). Trước đó sinh ảnh YOLO / eval YOLO chậm vì lý do này.
- **VLA**: camera cổ tay 2 tay (`L_wrist_cam`, `R_wrist_cam`), `homehand collect-vla` → 568 demo / 216k khung /
  200 ep (3 camera 256², 26 state/action, 4 cách diễn đạt câu lệnh + 1 giữ lại để test);
  `homehand/data/export_lerobot.py` (chạy bằng `.venv-vla`, LeRobot 0.6.1, H.264 + streaming encoding — AV1 mặc
  định quá chậm: 4 h) → 320 demo cho Kaggle; `kaggle/smolvla_finetune.ipynb` + `kaggle/README.md` (T4, 10k bước);
  `policy/smolvla_skill.py` + `scripts/eval_smolvla.py` (đã thử end-to-end với checkpoint 2 bước: ~97 ms/bước).
- **SmolVLA trên Kaggle** (notebook `notebookcf90b4f805`, dataset `homehand-vla-v1` riêng tư): T4 không có bf16 →
  VLM bf16 chạy giả lập > 7 s/bước (lần chạy 10k bị huỷ); vá `smolvlm_with_expert.py` sang float32 + `use_amp=true`
  → 1.4 s/bước; 3 000 bước (≈ 70 phút, loss 2.9 → 0.045). Model ở `models/smolvla_homehand` (đã vá float32 cả ở
  `.venv-vla`). Đánh giá (oracle, nominal, 10 ep): **0/10 sạch, 2/40 vật, 3 dừng an toàn** (lực tới 740 N ấn bàn);
  câu lệnh diễn đạt mới (5 ep): 3/20 vật, 1 dừng an toàn.
  Kết luận: pipeline chạy end-to-end, policy chưa đủ train (cần ≥ 20k bước ≈ 8 h T4). Zip phải tạo bằng Python
  (PowerShell 5.1 `Compress-Archive` ghi `\` → Kaggle từ chối).
- Windows + LeRobot: repo id thành `lerobot\smolvla_base` → tải về `models/smolvla_base`; symlink `last` lỗi
  quyền (chỉ Windows); torchcodec không nạp được → backend pyav; tên camera phải là camera1..3 (smolvla_base).

## 5. Con số hiện tại (phiên 5, Windows, 2026-10-10 — chạy lại toàn bộ bằng `scripts/run_all.py`, ~11 giờ)

Chi tiết đầy đủ: `RESULTS.md`. YOLO perception trừ khi ghi khác, 4 vật/episode (~một nửa vật ô ngoài nằm), CI Wilson 95%.

| Controller | nominal (50 ep.) sạch / vật | low / medium / high (25 ep.) vật | dừng an toàn (tổng) |
|---|---|---|---|
| Expert | 66% [52–78] / 87% [82–91] | 87% / 88% / 86% | 3 |
| Expert + oracle mask | 76% [63–86] / 92% [88–95] | – | 1 |
| ACT (25k bước, val MSE 0.0106) | 64% [50–76] / 86% [80–90] | 90% / 93% / 91% | 1 |
| Diffusion (30k bước, 30 DDIM, val MSE 0.041) | 30% [19–44] / 70% [63–76] | 69% / 68% / 62% | 18 |

- Grasp-bench (ground truth, 144 lần, đứng + nằm): **143/144** (1 mustard nằm trượt, tay phải).
- Motion quality (24 ep., ground truth): 18/24 sạch, 88/96 vật, 0 dừng an toàn, gia tốc khớp RMS 2.0 rad/s², lún tối đa 2 mm.
- Sim-gap từng yếu tố (16 ep./yếu tố): nominal 61/64; ma sát ×0.5 **52/64** (hại nhất, chủ yếu làm đổ vật);
  nhiễu pose 12 mm 56/64; độ cứng ngón ×0.6 58/64; ma sát ×1.5 59/64; khối lượng ×0.5/×2 60/59; trễ 120 ms 61/64.
- YOLO: mAP50 0.995 (mask mAP50-95 0.970), sai số vị trí 3.5 mm (p90 5.4), đứng/nằm đúng 97.9%.
- Dataset expert_v1: 870 demo / 324k khung (mỗi vật 196–236; tay trái 547, phải 323).

Nhận xét: expert thấp hơn bản cũ chỉ-vật-đứng (90%/98%) vì thêm vật nằm; lỗi chính trong cảnh đầy đủ là
knocked_over / dropped, dù từng cú nắm riêng lẻ gần như luôn thành công → va chạm giữa vật khi gắp/mang.
ACT ngang expert ở mọi mức (CI chồng nhau). Diffusion kém rõ và là controller duy nhất hay bị dừng an toàn (lực > 80 N).
Số cũ (máy Linux, phiên 1–4) không còn dùng; `data/archive/` trên máy Linux.

## 6. Trạng thái (2026-10-10 sáng)

- Pipeline đầy đủ đã chạy xong trên Windows (03:02), không lỗi. `data/` + `models/` trên máy Windows (không trong git).
- Thay đổi phiên 5 được commit vào nhánh `windows-port` (chưa push, chưa merge vào `main`).

## 7. Việc còn lại

1. Giảm knocked_over/dropped của expert trong cảnh đầy đủ (vd. chừa khoảng cách khi tay đi ngang vật bên cạnh,
   thứ tự gắp theo nguy cơ va chạm) — grasp riêng lẻ đã ~99%.
2. SmolVLA: train tiếp ≥ 20k bước (resume từ checkpoint trên Kaggle, ~8 h T4) rồi đánh giá lại; Diffusion Policy
   cũng cần train thêm.
3. Expert v2 còn chậm hơn người ~4× và đi vòng cao; thu demo / train lại ACT bằng expert v2.
4. Viewer trong Docker (X11) chưa thử; lớp `inspire_real.py` chưa thử trên tay thật.
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
