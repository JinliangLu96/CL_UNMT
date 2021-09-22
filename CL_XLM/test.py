# coding = utf-8
import numpy as np
import torch

if __name__ == "__main__":
    a = torch.from_numpy(np.array([
        [0.48, 0.77, 0.99, 0.28],
        [0.49, 0.59, 0.38, 0.57],
        [0.56, 0.78, 0.23, 0.67],
    ]))

    b = torch.from_numpy(np.array([
        [0.48, 0.77, 0.99, 0.28],
        [0.49, 0.59, 0.38, 0.57],
        [0.56, 0.78, 0.23, 0.67],
    ]))

    print((a*b).sum(dim=0))
    print(a.matmul(b.transpose(0, 1)))

    #
    # mask = torch.from_numpy(np.array([
    #     [1, 1, 1, 0],
    #     [1, 1, 1, 1],
    #     [1, 1, 0, 0],
    # ]))
    #
    # mask_bool = torch.from_numpy(np.array([
    #     [False, False, False, True],
    #     [False, False, False, False],
    #     [False, False, True, True],
    # ]))
    #
    # b = torch.from_numpy(np.array([
    #     [0.48, 0.77, 0.99, 0.0],
    #     [0.49, 0.59, 0.38, 0.57],
    #     [0.56, 0.78, 0.0, 0.0],
    # ]))
    #
    # print(a)
    # a = a.masked_fill_(mask_bool, value=float('-inf'))
    # print(a)
    # print(a.softmax(dim=-1))
    # print(torch.from_numpy(np.array(
    #     [0.48, 0.77, 0.99])).softmax(dim=-1))


    # b = torch.from_numpy(np.array([
    #     [0.98, 0.75, 0.69, 0.78],
    #     [0.49, 0.59, 0.38, 0.87],
    #     [0.53, 0.68, 0.73, 0.67],
    # ]))
    #
    # mask = torch.from_numpy(np.array([
    #     [1, 1, 1, 0],
    #     [1, 1, 1, 1],
    #     [1, 1, 0, 0],
    # ]))
    #
    # c = torch.from_numpy(np.array([
    #     [1.0, 1.0, 1.0, 1.0],
    #     [1.0, 1.0, 1.0, 1.0],
    #     [1.0, 1.0, 1.0, 1.0],
    # ]))
    # len = torch.from_numpy(np.array(
    #     [4, 3, 2]
    # ))
    #
    # print(a.size())
    # print(b.size())
    # print(len.size())
    # # print((torch.softmax(a, dim=1) * b).mean(dim=1))
    # print(((b * mask * a).transpose(0, 1)).mean(dim=0))
    # print(((b * mask * (torch.softmax(a, dim=1)))).sum(dim=1))
