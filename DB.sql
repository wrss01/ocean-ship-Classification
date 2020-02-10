CREATE TABLE `tianchi_01_train` (
  `id` bigint(20) DEFAULT NULL,
  `x` double DEFAULT NULL,
  `y` double DEFAULT NULL,
  `v` double DEFAULT NULL,
  `d` bigint(20) DEFAULT NULL,
  `t` text CHARACTER SET latin1,
  `ty` text COLLATE utf8mb4_bin
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;


CREATE TABLE `tianchi_01_test` (
  `id` bigint(20) DEFAULT NULL,
  `x` double DEFAULT NULL,
  `y` double DEFAULT NULL,
  `v` double DEFAULT NULL,
  `d` bigint(20) DEFAULT NULL,
  `t` text
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
