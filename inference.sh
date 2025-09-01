
#测试
python p_skin_test_m.py
python p_skin_test_v.py
python transfer_skin.py

python p_skeleton_test_v_small_id.py
python p_skeleton_test_m_small_id.py

python p_skeleton_test_v_larger_id.py
python p_skeleton_test_m_larger_id.py

python post_process.py




#预处理数据
#预处理random pose training data
python gen_random_pose_train_batch.py --cls mixamo --total_workers 2 --worker_id 0 
python gen_random_pose_train_batch.py --cls mixamo --total_workers 2 --worker_id 1
python gen_random_pose_train_batch.py --cls vroid --total_workers 2 --worker_id 0
python gen_random_pose_train_batch.py --cls vroid --total_workers 2 --worker_id 1

#预处理track pose training data
python gen_track_pose_train_batch.py --cls mixamo --total_workers 2 --worker_id 0
python gen_track_pose_train_batch.py--cls mixamo --total_workers 2 --worker_id 1
python gen_track_pose_train_batch.py--cls vroid --total_workers 2 --worker_id 0
python gen_track_pose_train_batch.py --cls vroid --total_workers 2 --worker_id 1

#预处理test data
python process_test.py

#训练蒙皮
python train_skin_m.py
python train_skin_v.py

#训练骨骼 id < 10000
python train_skeleton_v.py
python train_skeleton_m.py

#训练骨骼 id < 10000
python train_skeleton_m.py
python train_skeleton_v.py


#训练骨骼 id > 10000
python train_skeleton_v_pcae.py
python train_skeleton.py









