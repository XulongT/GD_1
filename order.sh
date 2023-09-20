python vis_smpl_kpt.py --data_dir ./DATA_DIR/motions_smpl --smpl_path ./SMPL_DIR/SMPL_FEMALE.pkl --sequence_name 0cZwsJ1u2d0_01_321_1026.pkl #2 p
python vis_smpl_kpt.py --data_dir ./DATA_DIR/motions_smpl --smpl_path ./SMPL_DIR/SMPL_FEMALE.pkl --sequence_name 0dqp2VUTYEY_02_0_372.pkl #4p


python vis_smpl_mesh.py --data_dir ./DATA_DIR/motions_smpl --smpl_path ./SMPL_DIR/SMPL_FEMALE.pkl --sequence_name JAtBXSA4aLg_03_0_810.pkl

#read n_persons
python motion_reader.py --data_dir ./DATA_DIR/motions_smpl --smpl_path ./SMPL_DIR/SMPL_FEMALE.pkl --sequence_name 2EhlnKTJ1vo_02_513_1950.pkl
python motion_reader.py --data_dir ./DATA_DIR/motions_smpl_max --smpl_path ./SMPL_DIR/SMPL_FEMALE.pkl --sequence_name 2EhlnKTJ1vo_02_513_1950.pkl
python motion_reader.py --data_dir ./DATA_DIR/motions_smpl --smpl_path ./SMPL_DIR/SMPL_FEMALE.pkl --sequence_name 0dqp2VUTYEY_02_0_372.pkl