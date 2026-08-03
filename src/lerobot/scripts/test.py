from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "CelerySticks/tracingLineData_20260729_151917"

dataset = LeRobotDataset(repo_id=repo_id)

print(dataset.features)