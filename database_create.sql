-- CREATE DATABASE db_signlanguage;
-- USE db_signlanguage;
-- SET @@global.sql_mode= '';
-- select @@global.sql_mode from dual;

-- CREATE TABLE actions (
--     action_id INT AUTO_INCREMENT PRIMARY KEY,
--     action_name VARCHAR(100) NOT NULL
-- );

-- CREATE TABLE sequences (
--     sequence_id INT AUTO_INCREMENT PRIMARY KEY,
--     sequence_num char,
--     action_id INT,
--     FOREIGN KEY (action_id) REFERENCES actions(action_id)
-- );

-- CREATE TABLE keyframes(
--     keyframe_id INT AUTO_INCREMENT PRIMARY KEY,
--     keyframe_num char,
--     sequence_id INT,
--     FOREIGN KEY (sequence_id) REFERENCES sequences(sequence_id)
-- );

-- CREATE TABLE pose_landmarks(
-- 	pose_id INT AUTO_INCREMENT PRIMARY KEY,
--     x FLOAT,
--     y FLOAT,
--     z FLOAT,
--     visibility FLOAT,
--     keyframe_id INT,
--     foreign key (keyframe_id) references keyframes(keyframe_id)
-- );

-- CREATE TABLE face_landmarks(
-- 	face_id INT AUTO_INCREMENT PRIMARY KEY,
--     x FLOAT,
--     y FLOAT,
--     z FLOAT,
--     keyframe_id INT,
--     foreign key (keyframe_id) references keyframes(keyframe_id)
-- );

-- CREATE TABLE rh_landmarks(
-- 	rh_id INT AUTO_INCREMENT PRIMARY KEY,
--     x FLOAT,
--     y FLOAT,
--     z FLOAT,
--     keyframe_id INT,
--     foreign key (keyframe_id) references keyframes(keyframe_id)
-- );

-- CREATE TABLE lh_landmarks(
-- 	lh_id INT AUTO_INCREMENT PRIMARY KEY,
--     x FLOAT,
--     y FLOAT,
--     z FLOAT,
--     keyframe_id INT,
--     foreign key (keyframe_id) references keyframes(keyframe_id)
-- );

-- -- drop database db_signlanguage;


CREATE DATABASE db_signlanguage;
USE db_signlanguage;

CREATE TABLE actions (
    action_id INT AUTO_INCREMENT PRIMARY KEY,
    action_name VARCHAR(100) NOT NULL
);

CREATE TABLE sequences (
    sequence_id INT AUTO_INCREMENT PRIMARY KEY,
    sequence_num INT NOT NULL,
    action_id INT NOT NULL,
    FOREIGN KEY (action_id) REFERENCES actions(action_id) ON DELETE CASCADE
);

CREATE TABLE keyframes (
    keyframe_id INT AUTO_INCREMENT PRIMARY KEY,
    keyframe_num INT NOT NULL,
    sequence_id INT NOT NULL,
    FOREIGN KEY (sequence_id) REFERENCES sequences(sequence_id) ON DELETE CASCADE
);

CREATE TABLE pose_landmarks (
    pose_id INT AUTO_INCREMENT PRIMARY KEY,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    z FLOAT NOT NULL,
    visibility FLOAT NOT NULL,
    keyframe_id INT NOT NULL,
    FOREIGN KEY (keyframe_id) REFERENCES keyframes(keyframe_id) ON DELETE CASCADE
);

CREATE TABLE face_landmarks (
    face_id INT AUTO_INCREMENT PRIMARY KEY,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    z FLOAT NOT NULL,
    keyframe_id INT NOT NULL,
    FOREIGN KEY (keyframe_id) REFERENCES keyframes(keyframe_id) ON DELETE CASCADE
);

CREATE TABLE lh_landmarks (
    lh_id INT AUTO_INCREMENT PRIMARY KEY,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    z FLOAT NOT NULL,
    keyframe_id INT NOT NULL,
    FOREIGN KEY (keyframe_id) REFERENCES keyframes(keyframe_id) ON DELETE CASCADE
);

CREATE TABLE rh_landmarks (
    rh_id INT AUTO_INCREMENT PRIMARY KEY,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    z FLOAT NOT NULL,
    keyframe_id INT NOT NULL,
    FOREIGN KEY (keyframe_id) REFERENCES keyframes(keyframe_id) ON DELETE CASCADE
);
