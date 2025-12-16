import torch
import torch.nn as nn
from openfold.model.model import AlphaFold
from openfold.data import data_transforms
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.loss import AlphaFoldLoss, lddt_ca
from openfold.config import model_config
import numpy as np
import ml_collections as mlc
import os
import argparse


# config = model_config("model_1", True)

# 初始化模型
# model = AlphaFold(config=config)  # 你可能需要传递一个适当的配置对象，根据你的需求配置模型

FACTOR = int(os.environ.get('BATCH_SIZE', 10)) # batch size
warmup = int(os.environ.get('WARM_UP', 5))
max_iters = int(os.environ.get('MAX_ITERS', 10))
IF_PROFILE_IN_DETAIL = False
USE_DTR =  True if os.environ.get('DTR_ENABLE') == '1' else False
RECORD_MEM_SNAPSHOT = True if os.environ.get('RECORD_MEM_SNAPSHOT') == '1' else False
snapshot_filename = os.environ.get('SNAP_FILE_NAME')

monomer_consts = mlc.ConfigDict(
    {
        "model": "model_1_ptm",  # monomer:model_1_ptm, multimer: model_1_multimer_v3
        "is_multimer": False,  # monomer: False, multimer: True
        "chunk_size": 4,
        "batch_size": 2,
        "n_res": 22 * FACTOR,
        "n_seq": 13 * FACTOR,
        "n_templ": 3 * FACTOR,
        "n_extra": 17 * FACTOR,
        "n_heads_extra_msa": 8,
        "eps": 5e-4,
        # For compatibility with DeepMind's pretrained weights, it's easiest for
        # everyone if these take their real values.
        "c_m": 256,
        "c_z": 128,
        "c_s": 384,
        "c_t": 64,
        "c_e": 64,
        "msa_logits": 23,  # monomer: 23, multimer: 22
        "template_mmcif_dir": None  # Set for test_multimer_datamodule
    }
)

multimer_consts = mlc.ConfigDict(
    {
        "model": "model_1_multimer_v3",  # monomer:model_1_ptm, multimer: model_1_multimer_v3
        "is_multimer": True,  # monomer: False, multimer: True
        "chunk_size": 4,
        "batch_size": 2,
        "n_res": 22,
        "n_seq": 13,
        "n_templ": 3,
        "n_extra": 17,
        "n_heads_extra_msa": 8,
        "eps": 5e-4,
        # For compatibility with DeepMind's pretrained weights, it's easiest for
        # everyone if these take their real values.
        "c_m": 256,
        "c_z": 128,
        "c_s": 384,
        "c_t": 64,
        "c_e": 64,
        "msa_logits": 22,  # monomer: 23, multimer: 22
        "template_mmcif_dir": None  # Set for test_multimer_datamodule
    }
)

consts = monomer_consts 

def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    string += ' | fragmentation: {}'.format(
        (torch.cuda.max_memory_reserved() - torch.cuda.max_memory_allocated()) / torch.cuda.max_memory_reserved()
    )
    # print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
    #         flush=True)
    print(string, flush=True)

def random_asym_ids(n_res, split_chains=True, min_chain_len=4):
    from random import randint
    n_chain = randint(1, n_res // min_chain_len) if consts.is_multimer else 1

    if not split_chains:
        return [0] * n_res

    assert n_res >= n_chain

    pieces = []
    asym_ids = []
    final_idx = n_chain - 1
    for idx in range(n_chain - 1):
        n_stop = (n_res - sum(pieces) - n_chain + idx - min_chain_len)
        if n_stop <= min_chain_len:
            final_idx = idx
            break
        piece = randint(min_chain_len, n_stop)
        pieces.append(piece)
        asym_ids.extend(piece * [idx])
    asym_ids.extend((n_res - sum(pieces)) * [final_idx])

    return np.array(asym_ids).astype(np.float32) + 1

def random_template_feats(n_templ, n, batch_size=None):
    b = []
    if batch_size is not None:
        b.append(batch_size)
    batch = {
        "template_mask": np.random.randint(0, 2, (*b, n_templ)),
        "template_pseudo_beta_mask": np.random.randint(0, 2, (*b, n_templ, n)),
        "template_pseudo_beta": np.random.rand(*b, n_templ, n, 3),
        "template_aatype": np.random.randint(0, 22, (*b, n_templ, n)),
        "template_all_atom_mask": np.random.randint(
            0, 2, (*b, n_templ, n, 37)
        ),
        "template_all_atom_positions": 
            np.random.rand(*b, n_templ, n, 37, 3) * 10,
        "template_torsion_angles_sin_cos": 
            np.random.rand(*b, n_templ, n, 7, 2),
        "template_alt_torsion_angles_sin_cos": 
            np.random.rand(*b, n_templ, n, 7, 2),
        "template_torsion_angles_mask": 
            np.random.rand(*b, n_templ, n, 7),
    }
    batch = {k: v.astype(np.float32) for k, v in batch.items()}
    batch["template_aatype"] = batch["template_aatype"].astype(np.int64)

    if consts.is_multimer:
        asym_ids = np.array(random_asym_ids(n))
        batch["asym_id"] = np.tile(asym_ids[np.newaxis, :], (*b, n_templ, 1))

    return batch

def random_extra_msa_feats(n_extra, n, batch_size=None):
    b = []
    if batch_size is not None:
        b.append(batch_size)
    batch = {
        "extra_msa": np.random.randint(0, 22, (*b, n_extra, n)).astype(
            np.int64
        ),
        "extra_has_deletion": np.random.randint(0, 2, (*b, n_extra, n)).astype(
            np.float32
        ),
        "extra_deletion_value": np.random.rand(*b, n_extra, n).astype(
            np.float32
        ),
        "extra_msa_mask": np.random.randint(0, 2, (*b, n_extra, n)).astype(
            np.float32
        ),
    }
    return batch



def generate_mock_data(batch_size, num_residues, num_sequences, num_templates, num_extra):
    # Make sure to use the correct dimension for target_feat
    data = {
        "aatype": torch.randint(0, 21, (batch_size, num_residues)),  # residue indices
        "target_feat": torch.randn(batch_size, num_residues, config.model.input_embedder.tf_dim),  # Correct dimension from config
        "residue_index": torch.arange(num_residues).repeat(batch_size, 1),  # Consecutive indices
        "msa_feat": torch.randn(batch_size, num_sequences, num_residues, config.model.input_embedder.msa_dim),  # MSA features
        "seq_mask": torch.ones(batch_size, num_residues),  # Sequence mask
        "msa_mask": torch.ones(batch_size, num_sequences, num_residues),  # MSA mask
        "pair_mask": torch.ones(batch_size, num_residues, num_residues),  # Pair mask
        "extra_msa_mask": torch.ones(batch_size, num_extra, num_residues),  # Extra MSA mask
        "template_mask": torch.ones(batch_size, num_templates),  # Template level mask
        "template_aatype": torch.clamp(torch.randint(0, 22, (batch_size, num_templates, num_residues)), max=20),  # Template residue indices
        "template_all_atom_positions": torch.randn(batch_size, num_templates, num_residues, 37, 3),  # Template atom coordinates
        "template_all_atom_mask": torch.ones(batch_size, num_templates, num_residues, 37),  # Template atom coordinate mask
        "template_pseudo_beta": torch.randn(batch_size, num_templates, num_residues, 3),  # Pseudo-beta positions
        "template_pseudo_beta_mask": torch.ones(batch_size, num_templates, num_residues)  # Pseudo-beta mask
    }
    return data


def main(args):
    if USE_DTR:
        torch.init_dtb_manager()
        mem_budget = args.mem_budget
        if mem_budget > 0:
            torch.set_memory_budget(int(mem_budget * 1e10))

    n_seq = consts.n_seq
    n_templ = consts.n_templ
    n_res = consts.n_res
    n_extra_seq = consts.n_extra

    c = model_config(consts.model, train=True)
    c.model.evoformer_stack.no_blocks = 4  # no need to go overboard here
    c.model.evoformer_stack.blocks_per_ckpt = None  # don't want to set up
    # deepspeed for this test

    loss_func = AlphaFoldLoss(c.loss)
    model = AlphaFold(c).cuda()

    if USE_DTR:
        print("trying to warp all parameters.")     # 这里进行前必须得先初始化了dtb系统
        model._apply(lambda v: v.detach().checkpoint(True))
        print("successfully warp all parameters.")

    learning_rate = 1e-3
    eps = 1e-5
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        eps=eps
    )

    model.train()

    import time
    # import nvtx

    def mock_inputs():
        batch = {}
        tf = torch.randint(c.model.input_embedder.tf_dim - 1, size=(n_res,))
        batch["target_feat"] = nn.functional.one_hot(
            tf, c.model.input_embedder.tf_dim
        ).float()
        # target_feat_tensors = []
        # for _ in range(batch_size):
        #     tf = torch.randint(c.model.input_embedder.tf_dim - 1, size=(n_res,))
        #     target_feat = nn.functional.one_hot(
        #         tf, c.model.input_embedder.tf_dim
        #     ).float()
        #     target_feat_tensors.append(target_feat)
        # batch["target_feat"] = torch.stack(target_feat_tensors, dim=0)

        batch["aatype"] = torch.argmax(batch["target_feat"], dim=-1)
        batch["residue_index"] = torch.arange(n_res)

        batch["msa_feat"] = torch.rand((n_seq, n_res, c.model.input_embedder.msa_dim))
        t_feats = random_template_feats(n_templ, n_res)
        batch.update({k: torch.tensor(v) for k, v in t_feats.items()})
        extra_feats = random_extra_msa_feats(n_extra_seq, n_res)
        batch.update({k: torch.tensor(v) for k, v in extra_feats.items()})
        batch["msa_mask"] = torch.randint(
            low=0, high=2, size=(n_seq, n_res)
        ).float()
        batch["seq_mask"] = torch.randint(low=0, high=2, size=(n_res,)).float()
        batch.update(data_transforms.make_atom14_masks(batch)) 
        # batch.update(data_transforms.make_atom14_positions(batch))
        batch["no_recycling_iters"] = torch.tensor(2.)

        if consts.is_multimer:
            batch["asym_id"] = torch.as_tensor(random_asym_ids(n_res))
            batch["entity_id"] = batch["asym_id"].clone()
            batch["sym_id"] = torch.ones(n_res)
            batch["extra_deletion_matrix"] = torch.randint(0, 2, size=(n_extra_seq, n_res))

        add_recycling_dims = lambda t: (
            t.unsqueeze(-1).expand(*t.shape, c.data.common.max_recycling_iters)
        )
        batch = tensor_tree_map(add_recycling_dims, batch)

        to_cuda_device = lambda t: t.cuda()
        batch = tensor_tree_map(to_cuda_device, batch)

        return batch

    def mock_backward_extra(batch):
        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)
        # print('===================================================')

        batch["atom14_atom_exists"] = torch.randint(0, 2, (n_res, 14)).to("cuda:0")
        batch["atom14_gt_positions"] = torch.tensor(np.random.rand(n_res, 14, 3)).to("cuda:0")
        batch["atom14_alt_gt_positions"] = torch.tensor(np.random.rand(n_res, 14, 3)).to("cuda:0")
        batch["atom14_gt_exists"] = torch.tensor(np.random.randint(0, 2, (n_res, 14)).astype(np.float32)).to("cuda:0")
        batch["atom14_atom_is_ambiguous"] = torch.tensor(np.random.randint(0, 2, (n_res, 14)).astype(np.float32)).to("cuda:0")
        batch["atom14_alt_gt_exists"] = torch.tensor(np.random.randint(0, 2, (n_res, 14)).astype(np.float32)).to("cuda:0")

        batch["pseudo_beta"] = torch.tensor(np.random.rand(n_res, 3).astype(np.float32)).to("cuda:0")
        batch["pseudo_beta_mask"] = torch.tensor(np.random.randint(0, 2, (n_res,)).astype(np.float32)).to("cuda:0")
        

        batch["all_atom_mask"] = torch.tensor(np.random.randint(0, 2, (n_res, 37)).astype(np.float32)).to("cuda:0")
        batch["atom37_atom_exists"] = torch.tensor(np.random.randint(0, 2, (n_res, 37)).astype(np.float32)).to("cuda:0")
        batch["resolution"] = torch.tensor(np.array(1.0)).to("cuda:0")

        from scipy.spatial.transform import Rotation
        def random_affines_vector(dim):
            prod_dim = 1
            for d in dim:
                prod_dim *= d

            affines = np.zeros((prod_dim, 7)).astype(np.float32)

            for i in range(prod_dim):
                affines[i, :4] = Rotation.random(random_state=42).as_quat()
                affines[i, 4:] = np.random.rand(
                    3,
                ).astype(np.float32)

            return affines.reshape(*dim, 7)
        
        batch["backbone_affine_tensor"] = torch.tensor(random_affines_vector((n_res,))).to("cuda:0")
        batch["backbone_affine_mask"] = torch.tensor(np.random.randint(0, 2, (n_res,)).astype(np.float32)).to("cuda:0")
        
        from openfold.utils.rigid_utils import Rigid
        def affine_vector_to_4x4(affine):
            r = Rigid.from_tensor_7(affine)
            return r.to_tensor_4x4()
        batch["backbone_rigid_tensor"] = affine_vector_to_4x4(
                batch["backbone_affine_tensor"]
            )
        batch["backbone_rigid_mask"] = batch["backbone_affine_mask"]
        # batch["backbone_rigid_mask"] = torch.tensor(np.random.randint(0, 2, (n_res,)).astype(np.float32)).to("cuda:0")

        batch["all_atom_positions"] = torch.tensor(np.random.rand(n_res, 37, 3).astype(np.float32)).to("cuda:0")
        sidechain_batch = data_transforms.atom37_to_frames(batch)
        batch.update(sidechain_batch)

        batch["true_msa"] = torch.tensor(np.random.randint(0, 21, (n_seq, n_res))).to("cuda:0")
        batch["bert_mask"] = torch.tensor(np.random.randint(0, 2, (n_seq, n_res)).astype(np.float32)).to("cuda:0")

        batch["chi_mask"] = torch.tensor(np.random.randint(0, 2, (n_res, 4)).astype(np.float32)).to("cuda:0")
        batch["chi_angles"] = torch.tensor(np.random.rand(n_res, 4).astype(np.float32)).to("cuda:0")
        batch["chi_angles_sin_cos"] = torch.stack(
            [
                torch.sin(batch["chi_angles"]),
                torch.cos(batch["chi_angles"]),
            ],
            dim=-1,
        )
        
        batch["seq_length"] = torch.tensor(np.array([n_res] * n_res, dtype=np.int32)).to("cuda:0")

        return batch


    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    iter_times = []

    for _ in range(max_iters+warmup):
        batch = mock_inputs()
        # if USE_DTR:
        #     for k,v in batch.items():
        #         batch[k] = v.checkpoint() if isinstance(v, torch.Tensor) else v
        # print(batch)
        print('===================================================')
        torch.cuda.synchronize()
        batch_start_time = time.time()
        # with torch.no_grad():
        if IF_PROFILE_IN_DETAIL:
            torch.cuda.nvtx.range_push("Start Forward")
            torch.cuda.synchronize()
            forward_start_time = time.time()

        outputs = model(batch)

        if IF_PROFILE_IN_DETAIL:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            print("forward time: ", time.time() - forward_start_time)
        # print(outputs)
        # print('===================================================')

        batch = mock_backward_extra(batch)
        # if USE_DTR:
        #     for k,v in batch.items():
        #         if isinstance(v, torch.Tensor):
        #             if not v.is_checkpoint():
        #                 print(k, v.checkpoint().shape)
                    # if not v.is_checkpoint() and v.shape:
                    #     batch[k] = v.checkpoint()

        
        # print(batch)
        loss, loss_breakdown = loss_func(
            outputs, batch, _return_breakdown=True
        )

        if IF_PROFILE_IN_DETAIL:
            torch.cuda.nvtx.range_push("Start backward")
            torch.cuda.synchronize()
            loss_start_time = time.time()

        loss.backward()

        if IF_PROFILE_IN_DETAIL:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            print("backward time: ", time.time() - loss_start_time)

        if IF_PROFILE_IN_DETAIL:
            torch.cuda.nvtx.range_push("Start Optimizer Step")
            torch.cuda.synchronize()
            opt_start_time = time.time()
            
        optimizer.step()

        if IF_PROFILE_IN_DETAIL:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            print("optimizer time: ", time.time() - opt_start_time)
        
        optimizer.zero_grad()

        torch.cuda.synchronize()
        iter_time = time.time() - batch_start_time
        print(f"batch {_} time:", iter_time, 's')
        if _ >= warmup:
            iter_times.append(iter_time)

        # if USE_DTR:
            # del batch
            # del loss
            # del outputs
            # torch.cuda.empty_cache()

    report_memory("[training summary]")
    print("avg train time:", sum(iter_times)/max_iters, 's')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mem_budget", type=float, help="")
    args = parser.parse_args()
    if RECORD_MEM_SNAPSHOT:
        torch.cuda.memory._record_memory_history()
    main(args)
    if RECORD_MEM_SNAPSHOT:
        torch.cuda.memory._dump_snapshot(snapshot_filename)