# faceRecognition

## 更新日志
### 2021/10/15 
* 增加多次识别逻辑 -n-c 和录像 -r 和遇到高难度样本的保护机制 -p
* 简化了main()结构
* 增加人脸识别模块时，输出位置--check_config_sum()--result_text=...
* 检测参数全部打包在info_dict内，直接从字典调用即可
* 合成了人脸识别部分

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

# 添加branch
git branch [分支名]

# 做出修改(上传到branch新建的branch)
git push origin [分支名]
```
### Pull Request

* 上传新branch后，点```pull request```进行merge，
  如果不能```automatic merge```再进行讨论或```issue```

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
### 设置识别逻辑（测试x次(1-199)，超过p(0-1)为真才确认为真)
```
python test.py -n [x] -c [p]
```
### 启动高难度样本保护
```
python test.py -p
```

## 退出
* 保持video窗口置顶
* 按q结束程序
* 按p推出系统保护（仅-p时有用）
* 没退出就多按几次
  