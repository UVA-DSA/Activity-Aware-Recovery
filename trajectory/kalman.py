import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, dt, x0, P0, Q, R, u_control, obstacles, k=1.0):
        """
        Initializes the Extended Kalman Filter.
        
        Parameters:
        - dt: Time step
        - x0: Initial state estimate (4x1 vector: [x, y, vx, vy])
        - P0: Initial covariance estimate (4x4 matrix)
        - Q: Process noise covariance (4x4 matrix)
        - R: Measurement noise covariance (2x2 matrix)
        - u_control: Control input acceleration (2x1 vector: [ax, ay])
        - obstacles: List of obstacle positions [(x1, y1), (x2, y2), ...]
        - k: Potential field constant
        """
        self.dt = dt
        self.x_hat = x0  # State estimate
        self.P = P0      # Covariance estimate
        self.Q = Q       # Process noise covariance
        self.R = R       # Measurement noise covariance
        self.u_control = u_control  # Control input acceleration
        self.obstacles = obstacles  # List of obstacle positions
        self.k = k  # Potential field constant

    def potential_field_acceleration(self, position):
        """
        Computes the acceleration due to the potential field at a given position.
        
        Parameters:
        - position: Current position (2x1 vector: [x, y])
        
        Returns:
        - a_pot: Acceleration due to potential field (2x1 vector: [ax_pot, ay_pot])
        """
        a_pot = np.zeros(2)
        for obs in self.obstacles:
            delta_pos = position - np.array(obs)
            distance_sq = np.sum(delta_pos ** 2)
            if distance_sq == 0:
                continue  # Avoid division by zero
            force_direction = delta_pos / np.sqrt(distance_sq)
            force_magnitude = self.k / distance_sq
            a_pot += force_magnitude * force_direction
        return a_pot

    def potential_field_jacobian(self, position):
        """
        Computes the Jacobian matrix of the potential field acceleration with respect to position.
        
        Parameters:
        - position: Current position (2x1 vector: [x, y])
        
        Returns:
        - J_pot: Jacobian matrix (2x2 matrix)
        """
        J_pot = np.zeros((2, 2))
        for obs in self.obstacles:
            delta_pos = position - np.array(obs)
            distance_sq = np.sum(delta_pos ** 2)
            if distance_sq == 0:
                continue  # Avoid division by zero
            distance = np.sqrt(distance_sq)
            k_over_d4 = self.k / distance_sq ** 2
            term1 = np.eye(2) / distance
            term2 = np.outer(delta_pos, delta_pos) / distance ** 3
            J = k_over_d4 * (term1 - 2 * term2)
            J_pot += J
        return J_pot

    def f(self, x, u):
        """
        State transition function.
        
        Parameters:
        - x: Previous state estimate (4x1 vector)
        - u: Control input (2x1 vector)
        
        Returns:
        - x_pred: Predicted state (4x1 vector)
        """
        dt = self.dt
        a = u
        x_pred = np.zeros(4)
        x_pred[0] = x[0] + x[2] * dt + 0.5 * a[0] * dt ** 2
        x_pred[1] = x[1] + x[3] * dt + 0.5 * a[1] * dt ** 2
        x_pred[2] = x[2] + a[0] * dt
        x_pred[3] = x[3] + a[1] * dt
        return x_pred

    def F_jacobian(self, x, u):
        """
        Computes the Jacobian matrix of the state transition function with respect to the state.
        
        Parameters:
        - x: Previous state estimate (4x1 vector)
        - u: Control input (2x1 vector)
        
        Returns:
        - F: Jacobian matrix (4x4 matrix)
        """
        dt = self.dt
        position = x[:2]
        velocity = x[2:]
        a_pot = self.potential_field_acceleration(position)
        J_pot = self.potential_field_jacobian(position)
        # Total acceleration
        a_total = self.u_control + a_pot
        # Jacobian matrix
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 0] += 0.5 * dt ** 2 * J_pot[0, 0]
        F[0, 1] += 0.5 * dt ** 2 * J_pot[0, 1]
        F[1, 0] += 0.5 * dt ** 2 * J_pot[1, 0]
        F[1, 1] += 0.5 * dt ** 2 * J_pot[1, 1]
        F[2, 0] += dt * J_pot[0, 0]
        F[2, 1] += dt * J_pot[0, 1]
        F[3, 0] += dt * J_pot[1, 0]
        F[3, 1] += dt * J_pot[1, 1]
        return F

    def predict(self):
        """
        Prediction step of the EKF.
        """
        # Compute potential field acceleration at previous position
        position = self.x_hat[:2]
        a_pot = self.potential_field_acceleration(position)
        # Total control input
        u_k = self.u_control + a_pot
        # State prediction
        self.x_hat = self.f(self.x_hat, u_k)
        # Jacobian of the state transition function
        F_k = self.F_jacobian(self.x_hat, u_k)
        # Covariance prediction
        self.P = F_k @ self.P @ F_k.T + self.Q

    def update(self, z):
        """
        Update step of the EKF.
        
        Parameters:
        - z: Measurement vector (2x1 vector: [x_meas, y_meas])
        """
        # Measurement matrix
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        # Measurement prediction
        z_pred = H @ self.x_hat
        # Innovation
        y = z - z_pred
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        # State update
        self.x_hat = self.x_hat + K @ y
        # Covariance update
        I = np.eye(len(self.x_hat))
        self.P = (I - K @ H) @ self.P

    def get_state(self):
        """
        Returns the current state estimate.
        """
        return self.x_hat
