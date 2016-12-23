import sys
import commands
import subprocess

def cmd(cmd):
	return commands.getoutput(cmd)
#	p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#	p.wait()
#	stdout, stderr = p.communicate()
#	return stdout.rstrip()

#labels
dirs = cmd("ls "+sys.argv[1])
labels = dirs.splitlines()

#make directries
cmd("mkdir images")

#copy images and make train.txt
pwd = cmd('pwd') #currentdirectory
imageDir = pwd+"/images"
train = open('train.txt','w')
test = open('test.txt','w')
labelsTxt = open('labels.txt','w')

classNo=0
cnt = 0
#label = labels[classNo]
for label in labels:
	workdir = pwd+"/"+sys.argv[1]+"/"+label #[label] directory of given directory
	imageFiles = cmd("ls "+workdir+"/*.jpg") #file in the above directory
	images = imageFiles.splitlines() #devide by return code(ex:crlf?) in above string
	print(label)
	labelsTxt.write(label+"\n") #write label name
	startCnt=cnt
	length = len(images)
	for image in images:
		imagepath = imageDir+"/image%07d" %cnt +".jpg"
		cmd("cp "+image+" "+imagepath) #copy and rename
		if cnt-startCnt < length*0.75: #devide into test and train data
			train.write(imagepath+" %d\n" % classNo)
		else:
			test.write(imagepath+" %d\n" % classNo)
		cnt += 1
		print cnt
	
	classNo += 1

train.close()
test.close()
labelsTxt.close()
