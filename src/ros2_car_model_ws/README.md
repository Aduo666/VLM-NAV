# ros2_car_model_ws
本工作空间负责将机器人原始 URDF 模型转化为 Isaac Sim 环境下的 USD 格式资产，并实现基础控制。  

## 编译项目
``` bash
cd ros2_cacr_model_ws
colcon build
```
## 启动GUI界面
进入isaacsim安装目录，运行`./isaac-sim.selector.sh`即可启动应用选择器  
![selector](./img/selector.png)  
点击 START 运行 Isaac Sim 主应用程序。  
更多资料以及文档可以查看官方网站`https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html`
## 导入机器人模型
选择File-Import  

![alt text](./img/setting.png)  
选择ros2_car_model_ws/src/urdf/1557/1557_all.urdf,选择Moveable Base,并将fr_wheel,fl_wheel,fr_wheel,fl_wheel的target改为velocity。点击import即可完成导入。

![alt text](./img/import.png)

![alt text](./img/importurdf.png)

## 创建控制器
Isaacsim有三种工作流
+ GUI  
主要功能：用于构建和模拟虚拟世界的可视化、直观、专业工具。  
推荐用法：世界构建、组装机器人、附加传感器、使用 OmniGraph 进行可视化编程以及初始化 ROS 桥接。  

+ Extensions  
主要功能：异步运行以允许与场景交互，热重载以立即反映更改，自适应物理步长以实现实时模拟。  
推荐用法：测试 Python 代码片段、构建交互式 GUI、自定义应用程序模块以及实时敏感型应用程序。
+ Standalone Python  
主要特性：可控制物理和渲染步骤的执行时间，支持无头模式运行。  
推荐用途：强化学习的大规模训练、系统化世界生成与修改。  

三者各有其特性，这里使用GUI与Standalone Python方式进行控制  
### GUI
在开始之前首先需要创建一个简单的环境，这里选择只加入一个地面，点击`Create-Physics-GroundPlane`加载一个地面，选择机器人将其拖至地面上方。  

点击`Tools-Robotics-OmnigraphControllers-Differential Controller`为机器人创建一个差速控制器（也可选择其他控制器类型如阿克曼控制器）。  
Robot Prim选择机器人prim（ranger_with_xarm6）。  
轮半径与轮间距填0.09，0.364。Right joint name和Left joint name填写机器人轮关节fr_wheel与fl_wheel。  
勾选Use_keyboard_control可以选择使用键盘控制。点击ok即可完成创建。   

这样创建后控制发现机器人在控制时方向并不符合预期，乱拐弯，这是因为只控制了前两个轮子，而后轮是僵的，因此可以以上面同样的方式为后轮也创建一个差速控制器实现四轮控制或者为小车重新创建一个差速控制器。  

创建后即可通过WASD控制或者在右侧stage中的找到ActionGraph中的differential_controller，修改property中的AngularVelocity与LinearVelocity即可。  

![alt text](./img/dfcontrol.png)
### Python脚本
在已经安装 Isaac Sim pip 包的 conda 环境中运行`ros2_car_model_ws_ws/src/scripts/load_keyboard_control.py`即可直接启动 Isaac Sim，并实现加载机器人与场景以及键盘控制，命令如下：  
``` bash
conda activate vlmnav
python nav_ws/src/scripts/load_keyboard_control.py
```  

![alt text](./img/pyscript.png)
