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
1. ~~Add spatial depth renderer to Camera (solved)~~
2. ~~Add reward to GoalPlatform and Goals using colliders in MoveHuman code (solved)~~
3. ~~Test rewards in human environment (solved)~~
4. ~~Find out why resolution is so low when executing built exe file (solved)~~
5. Walls are flying up in the sky
6. ~~env.reset can only work once~~
7. need another move action function (currently the agents aren't restricted by nav-mesh). Currently we have been trying to add controller to the agent. However, the agent is moving randomly and loosing control.

### Tips:
In brain parameters, space size is set to 3, e.g., when the only vector is position; stacked vector is set to 2 if we only want to store one historical vector (another is current vector).
