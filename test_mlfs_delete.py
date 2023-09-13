import tenplex
import torch


def test():
    ckpt = torch.load("/data/marcel/tmp/torch.ckpt")

    ip = "localhost"
    port = 20010
    client = tenplex.mlfs_client.MLFSClient(ip, port)

    job_id = "delete"
    device_rank = 0
    path = f"/job/{job_id}/save/{device_rank}"
    client.save_traverse(ckpt, path)
    client.upload_txt(f"job/{job_id}/iter", str(50))

    num_files, num_dirs = client.delete(path)
    print(f"num files {num_files}")
    print(f"num dirs {num_dirs}")


if __name__ == "__main__":
    test()
