# scl-navigation

Author *Zhoutong Wang*  & *Jialu Tan*

This is a repo dedicated to the Senseable City Lab project-AI stations. The goal of this project is to use Deep Reinforcement Learning Agent to validate, simulate and predict the wayfinding behavior of human-being in complicated indoor environments. The project code here is the DRL section which requires an built virtual environment to be designed in Unity, utilizing Unity's own Machine Learning package (ml-agents).

### Unity setup how-to:
1. Go to https://github.com/Unity-Technologies/ml-agents and clone the entire repo to local. 
2. Copy project file to **ml-agents\UnitySDK\Assets\ML-Agents\Examples**
3. Update Unity to new version
4. Open NavigationTest project scene in Unity hub
5. Connect **MazeLearningBrain** in **Brains** folder  to **StationAcademy** object. Don't forget to tick **control** box on the right side of Brain parameter
6. Go to **Agent** Object, drag necessary elements to **WallJumpAgent** parameters, add 1/2 (First person & Overhead) cameras to it as well. 
7. Set the the walking and running (usually 5/10) in **FirstHumanController** Script in **Human** Object. Set all objects in **MoveHuman** script as well. 
8. To be continued...

### DRL guidelines:
1. To be added...

### Things-to-do:

Week 4:
1. ~~Add spatial depth renderer to Camera (solved)~~
2. ~~Add reward to GoalPlatform and Goals using colliders in MoveHuman code (solved)~~
3. ~~Test rewards in human environment (solved)~~
4. ~~Find out why resolution is so low when executing built exe file (solved)~~
5. ~~env.reset can only work once~~
6. ~~need another move action function (the agents aren't restricted by nav-mesh). Currently we have been trying to add controller to the agent. However, the agent is moving randomly and loosing control.~~

Week5:
1. Make sure elevators work. (jtan)
2. Walls are flying up in the sky (jtan)
3. Export an plan for visualization (jtan)
4. Make training multi-processing (zwang)
5. Visualize training results (zwang)
6. Train... (zwang)

### Tips:
In brain parameters, space size is set to 3, e.g., when the only vector is position; stacked vector is set to 2 if we only want to store one historical vector (another is current vector).

To migrate mlagents to another version please visit https://github.com/Unity-Technologies/ml-agents/blob/0.11.0/docs/Migrating.md

Also, please use Window-Package Manager to install post-processing to enable camera effects


### Updates:
Dec 22 -
  1. agent reset problem: by deactivating navmeshagent before manually moving the agent and then activating after moving
  2. changed agent turning angle and walking speed

Dec 24 - 
  1. solved depth-map-all-white problem, changed 'far' variable in Unity (solved the turning parmenantly to right problem)
  2. add velocity to input vector
  3. multi-agent found we need multiple agents and environments in Unity
