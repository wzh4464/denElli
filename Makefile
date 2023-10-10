###
 # File: /Makefile
 # Created Date: Tuesday October 10th 2023
 # Author: Zihan
 # -----
 # Last Modified: Tuesday, 10th October 2023 5:37:47 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###


# libELSDc: target shared in ELSDc
# cd libELSDc && make shared
ELSDc/libelsdc.so:
	make -C ELSDc shared
