
# Ball and Beam system Optimal Control 


### Description
You can read the task description [here](https://github.com/NicholasBaraghini/Ball-and-Beam-system-Optimal-Control/files/7913643/OPTCON2021.-Group.21.Ball.and.Beam.project.proposal.pdf)

### Report
You can read and download our report [here](https://github.com/NicholasBaraghini/Ball-and-Beam-system-Optimal-Control/files/8009846/OPTCON_report_.1.pdf)


### Task 1 - Trajectory Exploration
Choose two equilibria and define a step between these two configurations. Compute the optimal transition for the ball to move from one equilibrium to another exploiting the DDP algorithm.

See the results in [this jupyter notebook](https://github.com/NicholasBaraghini/Ball-and-Beam-system-Optimal-Control/blob/main/Project%20Tasks/TASK_1_OPTCON_singlestep_Grp21_Ball_and_Beam_Project.ipynb) for a single step reference trajectory.Or download and run it!

![result example in Task 1 (single-step)](https://user-images.githubusercontent.com/76887265/152637309-2804d4fb-8cf2-46f0-9086-03e04e94db1c.jpg)

See the results in [this jupyter notebook](https://github.com/NicholasBaraghini/Ball-and-Beam-system-Optimal-Control/blob/main/Project%20Tasks/TASK_1_OPTCON_multistep_Grp21_Ball_and_Beam_Project.ipynb) for a multiple steps reference trajectory.Or download and run it!

![result example in Task 1 (multi-step)](https://user-images.githubusercontent.com/76887265/152637326-668c6586-be69-4ea3-b21f-b54e193df37e.jpg)



### Task 2 - Trajectory Optimization
Define the reference (quasi) trajectory which the ball needs to follow exploitingthe DDP algorithm to compute the optimal trajectory

See the results in [this jupyter notebook](https://github.com/NicholasBaraghini/Ball-and-Beam-system-Optimal-Control/blob/main/Project%20Tasks/TASK_2_OPTCON_qs_traj_Grp21_Ball_and_Beam_Project.ipynb) for the DDP algorithm apllied on the quasi-static reference trajectory. Or download and run it!

#### Quasi Static optimal trajectory
![state_reference_opt_qs_traj](https://user-images.githubusercontent.com/76887265/152655930-786311ef-dfd4-457c-808e-4bb6c982d023.jpg)

See the results in [this jupyter notebook](https://github.com/NicholasBaraghini/Ball-and-Beam-system-Optimal-Control/blob/main/Project%20Tasks/TASK_2_OPTCON_traj_Grp21_Ball_and_Beam_Project.ipynb) for the DDP algorithm apllied on the refined reference trajectory. Or download and run it!

#### Refined optimal trajectory
![state_reference_opt_traj](https://user-images.githubusercontent.com/76887265/152655936-cdc49d01-8222-4975-abca-f4446bc6a78a.jpg)




### Task 3 - Trajectory Tracking
Linearizing the system dynamics about the (optimal) trajectory (x, u) computed in Task 2, exploit the LQR algorithm to define the optimal feedback controller to track this reference trajectory.

See the results in [this jupyter notebook](https://github.com/NicholasBaraghini/Ball-and-Beam-system-Optimal-Control/blob/main/Project%20Tasks/TASK_3_OPTCON_init_disturbed_Grp21_Ball_and_Beam_Project.ipynb) for the optimal trajectory tracking.Or download and run it!

#### Results without noise
![state_track](https://user-images.githubusercontent.com/76887265/152655804-93e86a13-061b-434b-948c-7aea5ddd021e.jpg)

See the results in [this jupyter notebook](https://github.com/NicholasBaraghini/Ball-and-Beam-system-Optimal-Control/blob/main/Project%20Tasks/TASK_3_OPTCON_noise_Grp21_Ball_and_Beam_Project.ipynb) for the optimal trajectory tracking with affected by white noise.Or download and run it!

#### Results with noise (white noise)
![state_track_noise](https://user-images.githubusercontent.com/76887265/152655809-91a73b2a-ffa4-41a9-8be7-07f998e4568c.jpg)

See the results in [this jupyter notebook](https://github.com/NicholasBaraghini/Ball-and-Beam-system-Optimal-Control/blob/main/Project%20Tasks/TASK_3_OPTCON_traj_Grp21_Ball_and_Beam_Project.ipynb) for the tracking of the refined optimal trajectory.Or download and run it!




### Animations
#### Task 1 (single step)
![Task_1](https://user-images.githubusercontent.com/76887265/152637823-37191b5f-22f0-48da-a62b-400e1c066bf6.gif)

#### Task 1 (multi-step)
![Task_1_2](https://user-images.githubusercontent.com/76887265/152638321-556cf99f-5b52-4d22-a843-bbaaf5055d90.gif)

#### Task 2 (quasi-stationary trajectory)
![Task_2_qs_traj](https://user-images.githubusercontent.com/76887265/152655379-70e10030-dd35-43ef-8c77-49af63dd1256.gif)

#### Task 2 (refined trajectory)
![Task_2_traj](https://user-images.githubusercontent.com/76887265/152655393-a9563583-abd1-4e2a-9e59-7507462c7f25.gif)

#### Task 3 (without noise, initial consitions disturbed)
![Task_3_init_dist](https://user-images.githubusercontent.com/76887265/152655423-54a6b81c-91fc-405d-8b85-77495d0af5db.gif)

#### Task 3 (with noise)
![Task_3_noise](https://user-images.githubusercontent.com/76887265/152655442-06022776-1525-4ed4-aeb7-68bab92952d0.gif)


### Team

- [Nicholas Baraghini](https://github.com/NicholasBaraghini) 
- [Federico Iadarola](https://github.com/fedeiada)
- [Fabio Curto](https://github.com/FabioCurto)
