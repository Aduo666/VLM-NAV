from __future__ import annotations

class ContactMonitor:
    def __init__(self, runtime):
        from omni.physx import get_physx_simulation_interface
        from pxr import UsdPhysics

        self.runtime = runtime
        self.events = []
        self.wheels = {}
        joints = {prim.GetName(): UsdPhysics.Joint(prim) for prim in runtime.stage.Traverse()
                  if prim.IsA(UsdPhysics.Joint)}
        for wheel in runtime.profile.agx_controller.continuous.wheels:
            path = str(joints[wheel.drive_joint].GetBody1Rel().GetTargets()[0])
            self.wheels[path] = wheel.radius_m
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)

    @staticmethod
    def enable(stage, robot_path):
        from pxr import Usd, UsdPhysics, PhysxSchema
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)

    def _on_contact(self, headers, data):
        from pxr import PhysicsSchemaTools
        runtime = self.runtime
        if runtime.experiment is None or runtime.experiment.recorder is None:
            return
        root = runtime.profile.asset.prim_path + "/"
        for header in headers:
            if header.num_contact_data == 0:
                continue
            paths = [str(PhysicsSchemaTools.intToSdfPath(getattr(header, field)))
                     for field in ("actor0", "actor1", "collider0", "collider1")]
            robot_actors = [path for path in paths[:2] if path.startswith(root)]
            if len(robot_actors) != 1:
                continue
            actor = robot_actors[0]
            radius = self.wheels.get(actor)
            center_z = None
            if radius is not None:
                dc = runtime.drive._dc
                center_z = dc.get_rigid_body_pose(dc.get_rigid_body(actor)).p.z
            contacts = []
            for index in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                contact = data[index]
                position, normal = list(contact.position), list(contact.normal)
                support = (radius is not None and abs(normal[2]) >= 0.8
                           and position[2] < center_z - 0.5 * radius)
                if not support:
                    contacts.append(dict(position=position, normal=normal, separation_m=float(contact.separation),
                                         impulse=list(contact.impulse)))
            if contacts:
                self.events.append(dict(paths=paths, contacts=contacts, contact_timestamp_sec=runtime.timestamp))

    def close(self):
        self.subscription = None

