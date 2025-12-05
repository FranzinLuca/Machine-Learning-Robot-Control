#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformListener
import torch
import numpy as np
import joblib
import os
import json
import sys

sys.path.append(os.getcwd()) 
from src.model import DynamicMLP
from src.config import Config

class NeuralController(Node):
    def __init__(self, robot_name='Reacher3', model_dir='models'):
        super().__init__('neural_controller')
        
        # Configuration
        self.robot_name = robot_name
        self.declare_parameter('target_x', 0.5)
        self.declare_parameter('target_y', 0.5)
        self.declare_parameter('target_z', 0.5) 
        
        # Load Artifacts
        self.device = torch.device("cpu")
        self.load_artifacts(model_dir)
        
        # ROS Setup
        topic_name = '/arm_position_controller/commands'
        
        self.cmd_pub = self.create_publisher(Float64MultiArray, topic_name, 10)
        
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
            
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.current_q = None
        self.get_logger().info(f"Neural Controller for {robot_name} Ready! Publishing to {topic_name}")

    def load_artifacts(self, model_dir):
        # Load Config
        best_params = Config.best_hyperparameters(self.robot_name)
        self.n_layers = best_params['num_layers']
        self.hidden_layers = best_params['hidden_units']
        self.dropout = best_params['dropout']
        self.lr = best_params['learning_rate']
        self.activation = best_params['activation']
        self.use_residual = best_params['use_residuals']
        self.input_dim = 7 if self.robot_name == 'Reacher3' else (10 if self.robot_name == 'Reacher4' else 12)
        self.output_dim = 3 if self.robot_name == 'Reacher3' else (4 if self.robot_name == 'Reacher4' else 6)
        # Reconstruct Model
        self.model = DynamicMLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout,
            activation=self.activation,
            use_residual=self.use_residual
        )
        
        weights_path = os.path.join(model_dir, f'{self.robot_name}_best.pth')
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        
        self.x_scaler = joblib.load(os.path.join(model_dir, f'{self.robot_name}_final_x_scaler.pkl'))
        self.y_scaler = joblib.load(os.path.join(model_dir, f'{self.robot_name}_final_y_scaler.pkl'))

    def joint_callback(self, msg):
        if len(msg.position) >= self.output_dim:
            self.current_q = np.array(msg.position[:self.output_dim])

        
    def get_ee_position(self):
            try:
                t = self.tf_buffer.lookup_transform(
                    'base_link', 'tip_link', rclpy.time.Time())
                return np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
            except Exception as e:
                self.get_logger().warn(f"TF Failed: {e}", throttle_duration_sec=2.0)
                return None

    def control_loop(self):

        if self.current_q is None: return
        
        full_x = self.get_ee_position()
        if full_x is None: return

        if self.current_q is None:
            self.get_logger().info("Waiting for joint_states...", throttle_duration_sec=2.0)
            return
        
        full_x = self.get_ee_position()
        if full_x is None:
            self.get_logger().info("Waiting for TF (base_link -> tip_link)...", throttle_duration_sec=2.0)
            return

        # Target Setup
        target_full = np.array([
            self.get_parameter('target_x').value,
            self.get_parameter('target_y').value,
            self.get_parameter('target_z').value
        ])
        
        # Handle 2D (Reacher3) vs 3D
        if self.robot_name == 'Reacher3':
            x = full_x[:2]
            target = target_full[:2]
        else:
            x = full_x
            target = target_full

        # Control Logic
        dt = target - x
        dist = np.linalg.norm(dt)
        
        if dist < 0.02:
            self.get_logger().info(f"Target Reached! Error: {dist:.3f}", throttle_duration_sec=2.0)
            return

        # Normalize dt
        max_step = 0.05
        if dist > max_step:
            dt = (dt / dist) * max_step

        # Inference
        try:
            input_features = np.concatenate([x, self.current_q, dt])
            input_scaled = self.x_scaler.transform(input_features.reshape(1, -1))
            
            with torch.no_grad():
                dq_scaled = self.model(torch.FloatTensor(input_scaled)).numpy()
            
            dq = self.y_scaler.inverse_transform(dq_scaled).flatten()
            
            # Apply
            q_new = self.current_q + dq
            
            # Publish
            msg = Float64MultiArray()
            msg.data = q_new.tolist()
            self.cmd_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = NeuralController(robot_name='Reacher6')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()