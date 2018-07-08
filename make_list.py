#coding=utf-8
#对一批训练数据，里面包含多个文件夹，每个文件夹下面存放的是相同类别的物体
# 根据这些文件夹生成列表、切分验证、训练集数据
import os
import shutil
import  random
#因为caffe中,不允许文件名中有空格,所有需要重命名去除空格
def stdrename(imgfiles):
	for l in imgfiles:
		x_list=l.split(' ')
		y = ''.join(x_list)
		if l!=y:
			print 'rename'
			os.rename(l,y)

def GetFileList(FindPath,FlagStr=[]):
	FileList=[]
	FileNames=os.listdir(FindPath)
	if len(FileNames)>0:
		for fn in FileNames:
			if len(FlagStr)>0:
				if IsSubString(FlagStr,fn):
					fullfilename=os.path.join(FindPath,fn)
					FileList.append(fullfilename)
			else:
				fullfilename=os.path.join(FindPath,fn)
				FileList.append(fullfilename)


	if len(FileList)>0:
		FileList.sort()

	return FileList

def spiltdata(path_root,valratio=0.15):
	classify_temp=os.listdir(path_root)
	classify_file=[]
	for c in classify_temp:
		classify_file.append(os.path.join(path_root,c))



	for f in classify_file:
		imgfiles=GetFileList(f)
		stdrename(imgfiles)#caffe 文件名不允许有空格
	for c in classify_temp:
		imgfiles=os.listdir(os.path.join(path_root,c))
		nval=int(len(imgfiles)*valratio)
		print nval
		imgfvals=imgfiles[:nval]
	#验证数据文件列表
		for j in imgfvals:
			if os.path.exists(os.path.join(path_root+'/'+'val',c)) is False:
				os.makedirs(os.path.join(path_root+'/'+'val',c))
			newname=os.path.join(path_root+'/'+'val',c)+'/'+j
			oldname=os.path.join(path_root,c)+'/'+j
			shutil.move(oldname,newname)
	#训练数据文件列表
		imgftrains=imgfiles[nval:]
		for j in imgftrains:
			if os.path.exists(os.path.join(path_root+'/'+'train',c)) is False:
				os.makedirs(os.path.join(path_root+'/'+'train',c))
			newname=os.path.join(path_root+'/'+'train',c)+'/'+j
			oldname=os.path.join(path_root,c)+'/'+j
			shutil.move(oldname,newname)



def writetrainlist(path_root):
	classify_temp=os.listdir(path_root)
	classify_file=[]
	for c in classify_temp:
		classify_file.append(os.path.join(path_root,c))
	for f in classify_file:
		imgfiles=GetFileList(f)
		stdrename(imgfiles)#caffe 文件名不允许有空格

	sorted(classify_file)
	strlist=''
	for i,f in enumerate(classify_file):
		imgfiles=GetFileList(f)
		for image in imgfiles:
			print image
			strlist+=image+' '+str(i)+'\n'



	txtlist=open(path_root+'.txt','w')
	txtlist.write(strlist)
	txtlist.close()
def write_semi_trainlist(path_root,unlabel_ratio=0.1):
	classify_temp = os.listdir(path_root)
	classify_file = []
	for c in classify_temp:
		classify_file.append(os.path.join(path_root, c))
	for f in classify_file:
		imgfiles = GetFileList(f)
		stdrename(imgfiles)  # caffe 文件名不允许有空格

	sorted(classify_file)
	print classify_file
	strlist = ''
	super_strlist=''
	for i, f in enumerate(classify_file):
		imgfiles = GetFileList(f)
		for j,image in enumerate(imgfiles):
			if j<len(imgfiles)*unlabel_ratio:
				strlist += image + ' ' + str(i) + '\n'
				super_strlist+=image + ' ' + str(i) + '\n'
			else:
				strlist += image + ' ' + str(-1) + '\n'

	txtlist = open(path_root + '.txt', 'w')
	txtlist.write(strlist)
	txtlist.close()

	txtlist = open(path_root + 'super.txt', 'w')
	txtlist.write(super_strlist)
	txtlist.close()


#spiltdata('data/white')
'''writetrainlist('../faceshape/data/train')
writetrainlist('../faceshape/data/val')'''


#write_semi_trainlist("data/race/train")
writetrainlist('data/white/val')
writetrainlist('data/white/train')

