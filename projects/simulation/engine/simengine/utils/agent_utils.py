# Modified by ADToolbox contributors, 2026: standalone simulation and scene conversion support.
import cv2, numpy as np
import imageio
import os
from tqdm import tqdm
from pathlib import Path
from simengine.engine.engine_utils import get_engine
SIMREPO_ROOT = os.getenv('SIMREPO_ROOT', os.path.abspath('.'))

class SceneDataSaver():
    def __init__(self, path=None):
        engine = get_engine()
        job_name = engine.global_config.get('job_name', 'navsafe_filtered')
        self.base_path = 'dbg/nuplan_synthetic_video_8s2hz' if path is None else path
        self.human_sensor_blobs_path = Path('data/raw/openscene-v1.1/sensor_blobs/test')
        data_output_dir = engine.global_config.get('data_output_dir', f'{SIMREPO_ROOT}/data_output')
        self.sensor_blobs_path = Path(data_output_dir) / 'sensor_blobs'
        os.makedirs(self.base_path, exist_ok=True)
        
    def save_scene_data(self, scene_token, openscene_info_list):
        """
        Save the data of a single scene.
        Args:
            scene_token: The unique identifier of the scene.
        """
        engine = get_engine()
        data_manager = engine.managers['data_manager']
        current_scene = engine.current_scene
        
        # Get all frames data of the current scene
        scene_data = list(data_manager.episode_data.values())
        
        # Create the scene directory
        scene_path = os.path.join(self.base_path, scene_token)
        pred_render_path = os.path.join(scene_path, 'pred_render')
        human_raw_path = os.path.join(scene_path, 'human_raw')
        
        os.makedirs(pred_render_path, exist_ok=True)
        os.makedirs(human_raw_path, exist_ok=True)
        
        # Process from the 3rd frame
        for idx, frame in enumerate(scene_data[4:]):
            # Process the predicted rendering image (CAM_F0)
            if 'CAM_F0' in frame['cams']:
                # Get the path of the predicted rendering image
                pred_image_path = self.sensor_blobs_path / frame['cams']['CAM_F0']['data_path']
                if os.path.exists(pred_image_path):
                    # Read the image
                    pred_image = cv2.imread(str(pred_image_path))
                    if pred_image is not None:
                        # Save the predicted rendering image
                        save_name = f"{idx:02d}.jpg"
                        cv2.imwrite(os.path.join(pred_render_path, save_name), pred_image)

        for idx, frame in enumerate(openscene_info_list[4:]):
             if 'CAM_F0' in frame['cams']:
                    human_image_path = self.human_sensor_blobs_path / frame['cams']['CAM_F0']['data_path']
                    if os.path.exists(human_image_path):
                        # Read the original image
                        human_image = cv2.imread(str(human_image_path))
                        if human_image is not None:
                            # Save the original image
                            save_name = f"{idx:02d}.jpg"
                            cv2.imwrite(os.path.join(human_raw_path, save_name), human_image)
    
    def save_all_scenes(self):
        """
        Save the data of all scenes.
        """
        engine = get_engine()
        data_manager = engine.managers['data_manager']
        
        # Get all scene tokens
        scene_tokens = data_manager.get_all_scene_tokens()  # This method needs to be implemented based on actual conditions
        
        for scene_token in tqdm(scene_tokens, desc="Saving scenes"):
            self.save_scene_data(scene_token)

class _WaypointPolyline:
    """Minimal polyline adapter for BEV trajectory rendering."""
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)

    def get_polyline(self, line_sample_interval=2):
        return self.points


class BEV_visualizer():
    """
    Draw the start and end points, planning route, and ego agent waypoints.
    """
    def __init__(self, path=None):
        self.images = []
        self.path = 'dbg/controller_ego' if path is None else path
        self.plot_flag = True
        os.makedirs(self.path, exist_ok=True)

    def draw(self, step):
        from simengine.components.maps.map_constants import MapTerrainSemanticColor
        
        engine = get_engine()
        ego_agent = engine.managers['agent_manager']._dynamic_agents['ego']
        map = engine.current_map
        self.scene_id = engine.managers['scenario_manager'].current_scene_id

        # initial route
        reference_route = (ego_agent.navigation.checkpoint_lanes 
                          if type(ego_agent.navigation).__name__ == 'EgoLaneNavigation'
                          else ego_agent.navigation.current_ref_lanes)
        
        # Use the same world-to-pixel scale for the map and every overlay.
        bev_pixels_per_meter = 4

        # Color encoding used below is converted by output_gif; these values are
        # selected so the saved video appears with the intended RGB colors.
        route_color = (80 / 255, 160 / 255, 1.0)
        plan_color = (0.0, 1.0, 0.0)
        history_color = (1.0, 1.0, 0.0)
        traffic_color = (1.0, 165 / 255, 0.0)
        ego_color = (1.0, 1.0, 1.0)

        # initial mask: keep the route only, without ambiguous start/end dots.
        mask = map.get_trajectory_map(
            start_points=[],
            end_points=[],
            traj_lanes=reference_route,
            center_point=[0, 0],
            trajectory_color=route_color,
            trajectory_thickness=3,
            pixels_per_meter=bev_pixels_per_meter,
        )

        if step == 0:
            image_path = f'{self.path}/initial_route.png'
            cv2.imwrite(image_path, (mask * 255).astype(np.uint8))
            self.images.append((mask * 255).astype(np.uint8))
            return

        # create current frame mask
        cur_mask = mask.copy()
        
        # draw ego vehicle
        ego_box = np.array([
            [-ego_agent.length/2, -ego_agent.width/2],
            [ego_agent.length/2, -ego_agent.width/2],
            [ego_agent.length/2, ego_agent.width/2],
            [-ego_agent.length/2, ego_agent.width/2],
        ])
        ego_box = ego_agent.convert_to_world_coordinates(ego_box)
        
        # draw ego vehicle and current route
        cur_mask = map.get_trajectory_map_with_box(
            box_polygons=[ego_box],
            start_points=[],
            end_points=[],
            traj_lanes=[],
            center_point=[0, 0],
            trajectory_color=ego_color,
            trajectory_thickness=1,
            pixels_per_meter=bev_pixels_per_meter,
            box_color=ego_color,
            semantic_map=cur_mask,
        )

        # Actual driven trajectory, from the simulator's state buffer.
        agent_manager = engine.managers['agent_manager']
        ego_buffer = agent_manager._trajectory_buffer.get('ego', {}).get('state', {})
        positions = ego_buffer.get('position')
        valid = ego_buffer.get('valid')
        if positions is not None and valid is not None:
            upto = min(step + 1, len(positions))
            valid_positions = positions[:upto][valid[:upto, 0] > 0]
            if len(valid_positions) >= 2:
                cur_mask = map.get_trajectory_map(
                    start_points=[],
                    end_points=[],
                    traj_lanes=[_WaypointPolyline(valid_positions)],
                    center_point=[0, 0],
                    trajectory_color=history_color,
                    trajectory_thickness=2,
                    pixels_per_meter=bev_pixels_per_meter,
                    semantic_map=cur_mask,
                )

        # batch process waypoints
        if hasattr(ego_agent, 'trajectory') and ego_agent.trajectory and self.plot_flag:
            waypoints = ego_agent.trajectory.waypoints
            if len(waypoints) > 0:
                # Draw the future plan as one line, not one start marker per point.
                cur_mask = map.get_trajectory_map(
                    start_points=[],
                    end_points=[],
                    traj_lanes=[_WaypointPolyline(waypoints)],
                    center_point=[0, 0],
                    trajectory_color=plan_color,
                    trajectory_thickness=3,
                    pixels_per_meter=bev_pixels_per_meter,
                    semantic_map=cur_mask,
                )

        # batch process other agents
        other_agents = [(agent_id, agent) for agent_id, agent in 
                        engine.managers['agent_manager'].all_agents.items() 
                        if agent_id != 'ego']
        
        if other_agents:
            agent_boxes = []
            agent_positions = []
            for _, agent in other_agents:
                agent_box = np.array([
                    [-agent.length/2, -agent.width/2],
                    [agent.length/2, -agent.width/2],
                    [agent.length/2, agent.width/2],
                    [-agent.length/2, agent.width/2],
                ])
                agent_boxes.append(agent.convert_to_world_coordinates(agent_box))
                agent_positions.append(agent.current_position)

            # draw all other agents
            cur_mask = map.get_trajectory_map_with_box(
                box_polygons=agent_boxes,
                start_points=[],
                end_points=[],
                traj_lanes=[],
                center_point=[0, 0],
                trajectory_color=traffic_color,
                trajectory_thickness=1,
                pixels_per_meter=bev_pixels_per_meter,
                box_color=traffic_color,
                semantic_map=cur_mask,
            )

        # save image
        cur_mask_uint8 = (cur_mask * 255).astype(np.uint8)
        legend = (
            ('EGO  white', ego_color),
            ('PLAN green', plan_color),
            ('HIST yellow', history_color),
            ('CAR  orange', traffic_color),
            ('ROUTE blue', route_color),
        )
        cv2.rectangle(cur_mask_uint8, (18, 18), (360, 178), (25, 25, 25), -1)
        cv2.rectangle(cur_mask_uint8, (18, 18), (360, 178), (255, 255, 255), 1)
        for idx, (label, color) in enumerate(legend):
            y = 46 + idx * 28
            draw_color = tuple(int(round(channel * 255)) for channel in color)
            cv2.line(cur_mask_uint8, (30, y - 6), (70, y - 6), draw_color, 4)
            cv2.putText(cur_mask_uint8, label, (82, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, draw_color, 1, cv2.LINE_AA)
        # image_path = f'{self.path}/step_{step}_{self.scene_id}.png'
        # cv2.imwrite(image_path, cur_mask_uint8)
        self.images.append(cur_mask_uint8)  # directly use the array in memory, avoid repeated reading

    def output_gif(self):
        imageio.mimsave(f'{self.path}/simulation_{self.scene_id}.gif', self.images, fps=2)
        if not self.images:
            return
        frames = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in self.images]
        height, width = frames[0].shape[:2]
        video_path = f'{self.path}/simulation_{self.scene_id}.mp4'
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 2, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()
    
    def clear(self):
        self.images = []

class Video_visualizer():
    def __init__(self, path=None):
        self.images = []
        engine = get_engine()
        job_name = engine.global_config.get('job_name', 'navsafe_filtered')
        self.path = 'dbg/video_visualizer/' + job_name if path is None else path
        data_output_dir = engine.global_config.get('data_output_dir', f'{SIMREPO_ROOT}/data_output')
        self.sensor_blobs_path = Path(data_output_dir) / 'sensor_blobs'
        os.makedirs(self.path, exist_ok=True)
        
        # Define camera names and positions
        self.camera_names = ['CAM_F0', 'CAM_B0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2']
        self.camera_positions = {
            'CAM_F0': (0, 1),  # Front camera in the middle position
            'CAM_B0': (2, 1),  # Back camera in the middle position
            'CAM_L0': (0, 0),  # Left camera
            'CAM_L1': (1, 0),
            'CAM_L2': (2, 0),
            'CAM_R0': (0, 2),  # Right camera
            'CAM_R1': (1, 2),
            'CAM_R2': (2, 2)
        }
    
    def generate_video(self):
        engine = get_engine()
        data_manager = engine.managers['data_manager']
        agent_manager = engine.managers['agent_manager']

        self.data = list(data_manager.episode_data.values())

        # Get the first frame to determine the image size
        first_frame = self.data[0]
        self.log_token = first_frame['log_token']
        first_image_path = self.sensor_blobs_path / first_frame['cams']['CAM_F0']['data_path']
        first_image = cv2.imread(str(first_image_path))
        single_height, single_width = first_image.shape[:2]

        # Create a 3x3 grid layout
        grid_height = single_height * 3
        grid_width = single_width * 3

        # Set the video writer
        output_path = os.path.join(self.path, f'camera_view_{self.log_token}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            2,  # 0.5 s per step
            (grid_width, grid_height)
        )
        BEV_image = None

        try:
            for idx, frame in enumerate(tqdm(self.data, desc="Processing frames")):
                # Create a blank grid
                grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                if agent_manager._BEV_vis is not None:
                    BEV_image = agent_manager._BEV_vis.images[idx]
                # Put each camera's image into the grid
                for cam_name in self.camera_names:
                    if cam_name.upper() in frame['cams']:  # Note the case
                        # Load the image from the file path
                        image_path = self.sensor_blobs_path / frame['cams'][cam_name.upper()]['data_path']
                        image = cv2.imread(str(image_path))
                        
                        if image is not None:
                            grid_row, grid_col = self.camera_positions[cam_name]
                            y_start = grid_row * single_height
                            x_start = grid_col * single_width
                            grid_image[y_start:y_start + single_height,
                                     x_start:x_start + single_width] = image

                            # Add the camera name label
                            cv2.putText(grid_image, cam_name.upper(),
                                      (x_start + 10, y_start + 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if BEV_image is not None:
                    bev_h, bev_w = BEV_image.shape[:2]
                    y0 = max(0, bev_h // 2 - single_height // 2)
                    x0 = max(0, bev_w // 2 - single_width // 2)
                    center_crop = BEV_image[y0:y0 + single_height, x0:x0 + single_width, :3]
                    if center_crop.shape[:2] != (single_height, single_width):
                        center_crop = cv2.resize(
                            center_crop, (single_width, single_height),
                            interpolation=cv2.INTER_LINEAR)
                    grid_image[single_height:single_height*2, single_width:single_width*2] = center_crop
                    
                    # Add command text
                    if 'command' in frame:
                        command = frame['command']
                        command_text = f"Command: {command}"
                        cv2.putText(grid_image, command_text,
                                  (single_width + 10, single_height + 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                video_writer.write(grid_image)

        finally:
            video_writer.release()
            print(f"Video saved to {output_path}")

def get_current_lane(agent, map):
    """
    Used to find current lane information.
    Args:
        agent: BaseAgent, the agent whose current lane is to be found.
        map: Map, the map containing the road network.

    Returns:
        tuple: A tuple containing the dist, lane index, and lane object.
    """
    if agent.length is not None:
        lane_valid_range = agent.length
    else:
        lane_valid_range = 10.0

    possible_lanes = map.road_network.get_closest_lane_index(
        agent.current_position, return_all=True)

    valid_possible_lanes = [
        (dist, lane_index, lane)
        for dist, lane_index, lane in possible_lanes if dist < lane_valid_range
    ]

    valid_possible_lane_indexes = [
        lane_index for dist, lane_index, lane in valid_possible_lanes
    ]

    if agent.navigation.next_ref_lanes is not None:
        current_ref_lanes = agent.navigation.current_ref_lanes
        next_ref_lanes = agent.navigation.next_ref_lanes
        target_checkpoints_index = agent.navigation._target_checkpoints_index
        checkpoints_lane_indexes = agent.navigation.checkpoints_lane_indexes

        # Check if the lane is in current reference lanes
        for idx, lane_index in enumerate(valid_possible_lane_indexes):
            if lane_index in current_ref_lanes:
                return valid_possible_lanes[idx]

        # Check if the vehicle has reached the destination or no further lanes to move
        nx_ckpt_index = target_checkpoints_index[-1]
        if (nx_ckpt_index == checkpoints_lane_indexes[-1]  # next checkpoint is the last
                or next_ref_lanes is None):  # no lanes for the vehicle to move.
            return valid_possible_lanes[0]

        # Check if the vehicle has moved to the next reference lanes
        for idx, lane_index in enumerate(valid_possible_lane_indexes):
            if lane_index in next_ref_lanes:
                return valid_possible_lanes[idx]
        
        return valid_possible_lanes[0]
    
    else :
        # If no lane is found in the next_ref lanes, find the best lane via heading_diff
        min_heading_diff = float('inf')
        best_lane = None
        best_lane_info = None

        for lane_info in valid_possible_lanes:
            lane = lane_info[2]
            long, _ = lane.local_coordinates(agent.current_position)
            lane_heading = lane.heading_theta_at(long)
            heading_diff = abs(agent.current_heading - lane_heading)

            if heading_diff < min_heading_diff:
                min_heading_diff = heading_diff
                best_lane = lane
                best_lane_info = lane_info

        if best_lane is None:
            return None, None, None

        return best_lane_info
