CREATE DATABASE db_signlanguage;
USE db_signlanguage;

CREATE TABLE actions (
    action_id INT AUTO_INCREMENT PRIMARY KEY,
    action_name VARCHAR(100) NOT NULL
);

CREATE TABLE sequences (
    sequence_id INT AUTO_INCREMENT PRIMARY KEY,
    sequence_num char,
    action_id INT,
    FOREIGN KEY (action_id) REFERENCES actions(action_id)
);

CREATE TABLE keyframes(
    keyframe_id INT AUTO_INCREMENT PRIMARY KEY,
    keyframe_num char,
    sequence_id INT,
    FOREIGN KEY (sequence_id) REFERENCES sequences(sequence_id)
);

CREATE TABLE pose_landmarks(
	pose_id INT AUTO_INCREMENT PRIMARY KEY,
    x FLOAT,
    y FLOAT,
    z FLOAT,
    visibility FLOAT,
    keyframe_id INT,
    foreign key (keyframe_id) references keyframes(keyframe_id)
);

CREATE TABLE face_landmarks(
	face_id INT AUTO_INCREMENT PRIMARY KEY,
    x FLOAT,
    y FLOAT,
    z FLOAT,
    keyframe_id INT,
    foreign key (keyframe_id) references keyframes(keyframe_id)
);

CREATE TABLE rh_landmarks(
	rh_id INT AUTO_INCREMENT PRIMARY KEY,
    x FLOAT,
    y FLOAT,
    z FLOAT,
    keyframe_id INT,
    foreign key (keyframe_id) references keyframes(keyframe_id)
);

CREATE TABLE lh_landmarks(
	lh_id INT AUTO_INCREMENT PRIMARY KEY,
    x FLOAT,
    y FLOAT,
    z FLOAT,
    keyframe_id INT,
    foreign key (keyframe_id) references keyframes(keyframe_id)
);

-- drop database db_signlanguage;