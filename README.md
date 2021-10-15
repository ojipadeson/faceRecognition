# faceRecognition

## 更新日志
### <日期> <内容>

## Install
### 配置环境
```
pip install -r requirements.txt
```
### 克隆到本地
```
git clone https://github.com/ojipadeson/faceRecognition
```

## Work on faceRecognition
### 本地修改上传
```
# 做修改并将文件归档 (对每一个要修改的文件执行'git add'或执行'git add .'归档所有文件)
git add <文件名>

# 提交代码
git commit -m "附上的评论"

# 对master（默认）做出修改(建议上传到branch)
git push origin [分支名]
git push origin master
```
### 添加和删除branch
```
# 添加
git branch [分支名]
# 删除
git branch -d [分支名]
```
### 更新本地
```
git pull https://github.com/ojipadeson/faceRecognition
```

## 运行
### 按默认运行
```
python test.py
```
### 保存运行录像
```
python test.py -r
```
### 设置识别逻辑（测试x次，超过80%为真才确认为真)
```
python test.py -n [x]
```
