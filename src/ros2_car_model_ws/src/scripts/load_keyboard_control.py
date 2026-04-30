# ==========================================================
#  Isaac Sim 4.5 — 载入 URDF 车辆并控制运动
#
# ==========================================================

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # GUI 模式运行

# ---------- 基础依赖 ----------
import omni
import carb
import time
import numpy as np
import os
from pxr import Gf, UsdGeom
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

from isaacsim.storage.native import get_assets_root_path
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.api.objects.ground_plane import GroundPlane
from omni.isaac.dynamic_control import _dynamic_control


# ---------- 路径与参数 ----------
#将/home/gm_2774373792/ros2_car_model/ros2_car_model_ws/替换为自己工作空间的路径
urdf_path = "/home/gm_3392345164/ros2_car_model_ws/src/1557/urdf/1557_all.urdf"
base_dir = "/World/ranger_with_xarm6/joints"
vel = 10
angel = 0.5

# ---------- 初始化 Stage ----------
stage = omni.usd.get_context().get_stage()
if not stage:
    raise RuntimeError("无法获取场景 Stage，请检查 SimulationApp 是否初始化")

# 创建默认 Prim "/World"
if not stage.GetPrimAtPath("/World"):
    world_prim = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world_prim.GetPrim())
    print(" 已创建 /World 作为 defaultPrim")
else:
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    print(" 已存在 /World，设置为 defaultPrim")

#  添加基础灯光
# 方向光（模拟太阳光）
sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
sun.CreateIntensityAttr(5000.0)
sun.CreateAngleAttr(0.53)
sun.CreateColorAttr((1.0, 0.98, 0.92))
print(" 添加方向光 /World/SunLight")

# 加载地面
GroundPlane(prim_path="/World/GroundPlane", z_position=0.0)

# ---------- 导入 URDF 模型 ----------
urdf_interface = _urdf.acquire_urdf_interface()
import_config = _urdf.ImportConfig()
import_config.fix_base = False
import_config.make_default_prim = False
import_config.self_collision = False
import_config.convex_decomp = False
import_config.distance_scale = 1.0
import_config.density = 0.0

print(" 正在导入 URDF 模型...")
result, robot_model = omni.kit.commands.execute(
    "URDFParseFile",
    urdf_path=urdf_path,
    import_config=import_config,
)
result, prim_path = omni.kit.commands.execute(
    "URDFImportRobot",
    urdf_robot=robot_model,
    import_config=import_config,
)
print(f" 模型已加载到 {prim_path}")

def place_on_ground(prim_path):
    """
    将车辆放置在地面 z=0 上
    """
    car_prim = stage.GetPrimAtPath(prim_path)
    if not car_prim:
        print(f" 未找到车辆 Prim: {prim_path}")
        return

    # 遍历所有子 Prim，找到最底部的点
    bbox_min_z = None
    def traverse(prim):
        nonlocal bbox_min_z
        geom = UsdGeom.Xformable(prim)
        if geom:
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(prim)
            if bbox:
                min_z = bbox.GetRange().GetMin()[2]
                if bbox_min_z is None or min_z < bbox_min_z:
                    bbox_min_z = min_z
        for child in prim.GetChildren():
            traverse(child)

    traverse(car_prim)

    if bbox_min_z is None:
        print(" 未能获取车辆最低点，保持原始位置")
        return

    # 计算 z 偏移量
    z_offset = -bbox_min_z
    xform = UsdGeom.Xformable(car_prim)

    #  获取已有 translateOp
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp()

    # 设置新的 z 偏移量
    old_val = translate_op.Get()
    new_val = (old_val[0], old_val[1], old_val[2] + z_offset)
    translate_op.Set(new_val)
    print(f" 车辆已放置在地面上，z 偏移量 = {z_offset:.3f}")

# 调用函数，将车辆放到地面
place_on_ground(prim_path)

# ---------- 启动时间线并开始仿真 ----------
timeline = omni.timeline.get_timeline_interface()
timeline.play()
print(" 开始仿真...")
simulation_app.update()

# ---------- 自动检测 Articulation Root ----------
dc = _dynamic_control.acquire_dynamic_control_interface()
articulation_root = None
for prim in stage.TraverseAll():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        articulation_root = prim
        print(f" 找到 Articulation Root : {articulation_root.GetPath()}")
        break

if articulation_root is None:
    raise RuntimeError(" 未找到 Articulation Root，请检查 URDF 是否导入成功")

# === 打印所有 DOF 名称 ===
dc = _dynamic_control.acquire_dynamic_control_interface()
articulation = dc.get_articulation(str(articulation_root.GetPath()))
print(" 该 articulation 中的 DOFs：")
for i in range(dc.get_articulation_dof_count(articulation)):
    dof_handle = dc.get_articulation_dof(articulation, i)
    dof_name = dc.get_dof_name(dof_handle)
    print(f"  {i}: {dof_name}")
    
articulation = dc.get_articulation(str(articulation_root.GetPath()))
dc.wake_up_articulation(articulation)

# ---------- 车辆控制函数 ----------
def wheel_init(wheel_name, stiffness, damping):
    path = os.path.join(base_dir, wheel_name)
    wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(path), "angular")
    if wheel_drive:
        wheel_drive.GetStiffnessAttr().Set(stiffness)
        wheel_drive.GetDampingAttr().Set(damping)
    else:
        print(f" 未找到 DriveAPI 路径 : {path}")

def wheel_drive(wheel_name, vel, flag):
    dof_wheel = dc.find_articulation_dof(articulation, wheel_name)
    if flag == "backward":
        vel = -vel
    dc.set_dof_velocity_target(dof_wheel, vel)

def wheel_rotate(wheel_steering_name, angle, flag):
    dof_str = dc.find_articulation_dof(articulation, wheel_steering_name)
    if flag == "left":
        angle = -angle
    dc.set_dof_position_target(dof_str, angle)
    
def wheel_reset():
    for s in ["fl_steering_joint", "fr_steering_joint"]:
        wheel_rotate(s, 0, "left")
    for w in ["fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"]:
        wheel_drive(w, 0, "forward")
        
def car_forward():
    wheel_reset()
    for w in ["fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"]:
        wheel_drive(w, vel, "forward")

def car_backward():
    wheel_reset()
    for w in ["fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"]:
        wheel_drive(w, vel, "backward")

def turn_left():
    
    for s in ["fl_steering_joint", "fr_steering_joint"]:
        wheel_rotate(s, angel, "left")

def turn_right():
    
    for s in ["fl_steering_joint", "fr_steering_joint"]:
        wheel_rotate(s, angel, "right")

# ---------- 初始化车轮驱动 ----------
for w in ["fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"]:
    wheel_init(w, 0, 10000)
    
    
input_interface = carb.input.acquire_input_interface()
app_window = omni.appwindow.get_default_app_window()
keyboard_handle = app_window.get_keyboard()

# 定义按键命令状态
key_state = {
    "W": False,
    "S": False,
    "A": False,
    "D": False
}

# 键盘事件回调函数
def _keyboard_event_cb(event, *args, **kwargs):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
        if event.input == carb.input.KeyboardInput.W:
            key_state["W"] = True
        elif event.input == carb.input.KeyboardInput.S:
            key_state["S"] = True
        elif event.input == carb.input.KeyboardInput.A:
            key_state["A"] = True
        elif event.input == carb.input.KeyboardInput.D:
            key_state["D"] = True
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        if event.input == carb.input.KeyboardInput.W:
            key_state["W"] = False
        elif event.input == carb.input.KeyboardInput.S:
            key_state["S"] = False
        elif event.input == carb.input.KeyboardInput.A:
            key_state["A"] = False
        elif event.input == carb.input.KeyboardInput.D:
            key_state["D"] = False

# 订阅键盘事件
input_interface.subscribe_to_keyboard_events(keyboard_handle, _keyboard_event_cb)

print(" 键盘控制已启用：W 前进  S 后退  A 左转  D 右转")

# 主循环：实时控制
while True:
    simulation_app.update()
   
    time.sleep(0.1)
    
    # 根据按键状态控制车辆
    if key_state["W"]:
        car_forward()
    elif key_state["S"]:
        car_backward()
    else:
        wheel_reset()
    
    if key_state["A"]:
        turn_left()
    elif key_state["D"]:
        turn_right()
  
    


    
#---------- 结束仿真 ----------
#timeline.stop()
#simulation_app.close()
