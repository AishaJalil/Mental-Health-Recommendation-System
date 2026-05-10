"""
Neural Collaborative Filtering (NCF) — NeuMF variant.
Combines Generalized Matrix Factorization (GMF) and an MLP branch.
Pure-NumPy implementation — no PyTorch or TensorFlow required.

Architecture:
  GMF branch : pu ⊙ qi  (element-wise product of user/item GMF embeddings)
  MLP branch : ReLU layers on concat(pu_m, qi_m)
  Output     : sigmoid( h · concat(gmf_out, mlp_out) )

Training    : binary cross-entropy with negative sampling (implicit feedback).
              Ratings ≥ 4 are treated as positive interactions.
"""

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NeuralCFRecommender:

    def __init__(
        self,
        gmf_factors: int = 16,
        mlp_factors: int = 16,
        mlp_hidden: tuple = (32, 16),
        lr: float = 0.005,
        n_epochs: int = 15,
        batch_size: int = 512,
        neg_ratio: int = 4,
        reg: float = 0.001,
    ):
        self.gmf_factors = gmf_factors
        self.mlp_factors = mlp_factors
        self.mlp_hidden  = mlp_hidden
        self.lr          = lr
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.neg_ratio   = neg_ratio
        self.reg         = reg

        self._user_idx: dict = {}
        self._item_idx: dict = {}
        self._idx_item: dict = {}
        self._products       = None
        self._trained        = False

        self._gmf_Pu = self._gmf_Qi = None  # GMF embeddings
        self._mlp_Pu = self._mlp_Qi = None  # MLP embeddings
        self._W: list = []                   # MLP layer weight matrices
        self._b: list = []                   # MLP layer bias vectors
        self._h: np.ndarray = None           # output weight vector

    # ------------------------------------------------------------------ #
    def fit(self, ratings_df, products_df):
        self._products = products_df.copy()

        users = ratings_df["user_id"].unique()
        items = ratings_df["product_id"].unique()
        self._user_idx = {u: i for i, u in enumerate(users)}
        self._item_idx = {it: i for i, it in enumerate(items)}
        self._idx_item = {i: it for it, i in self._item_idx.items()}

        n_u, n_i = len(users), len(items)
        rng = np.random.default_rng(42)
        s   = 0.01

        self._gmf_Pu = rng.standard_normal((n_u, self.gmf_factors)) * s
        self._gmf_Qi = rng.standard_normal((n_i, self.gmf_factors)) * s
        self._mlp_Pu = rng.standard_normal((n_u, self.mlp_factors)) * s
        self._mlp_Qi = rng.standard_normal((n_i, self.mlp_factors)) * s

        layer_sizes = [self.mlp_factors * 2] + list(self.mlp_hidden)
        self._W, self._b = [], []
        for in_sz, out_sz in zip(layer_sizes[:-1], layer_sizes[1:]):
            bound = np.sqrt(2.0 / in_sz)   # He initialisation
            self._W.append(rng.standard_normal((in_sz, out_sz)) * bound)
            self._b.append(np.zeros(out_sz))

        h_size = self.gmf_factors + self.mlp_hidden[-1]
        self._h = rng.standard_normal(h_size) * s

        # Positive pairs: user-item interactions with rating >= 4
        pos_rows = [
            (self._user_idx[r.user_id], self._item_idx[r.product_id])
            for r in ratings_df.itertuples()
            if r.rating >= 4
               and r.user_id    in self._user_idx
               and r.product_id in self._item_idx
        ]
        if not pos_rows:
            pos_rows = [
                (self._user_idx[r.user_id], self._item_idx[r.product_id])
                for r in ratings_df.itertuples()
                if r.user_id in self._user_idx and r.product_id in self._item_idx
            ]

        print(f"  NCF: {n_u} users | {n_i} items | {len(pos_rows)} positive pairs")

        for epoch in range(self.n_epochs):
            # Negative sampling — uniform random items (approximate)
            neg_u = np.repeat([p[0] for p in pos_rows], self.neg_ratio)
            neg_i = rng.integers(0, n_i, size=len(pos_rows) * self.neg_ratio)

            all_u = np.concatenate([[p[0] for p in pos_rows], neg_u]).astype(np.int32)
            all_i = np.concatenate([[p[1] for p in pos_rows], neg_i]).astype(np.int32)
            all_y = np.concatenate([
                np.ones(len(pos_rows)), np.zeros(len(neg_u))
            ])

            perm = rng.permutation(len(all_u))
            all_u, all_i, all_y = all_u[perm], all_i[perm], all_y[perm]

            epoch_loss, n_batches = 0.0, 0
            for start in range(0, len(all_u), self.batch_size):
                sl = slice(start, start + self.batch_size)
                epoch_loss += self._train_batch(all_u[sl], all_i[sl], all_y[sl])
                n_batches  += 1

            if (epoch + 1) % 5 == 0:
                avg = epoch_loss / max(n_batches, 1)
                print(f"    Epoch {epoch+1}/{self.n_epochs} — loss: {avg:.4f}")

        self._trained = True

    # ------------------------------------------------------------------ #
    def _forward(self, u_idx: np.ndarray, i_idx: np.ndarray):
        # GMF branch
        pu      = self._gmf_Pu[u_idx]
        qi      = self._gmf_Qi[i_idx]
        gmf_out = pu * qi                      # (B, gmf_factors)

        # MLP branch
        pu_m = self._mlp_Pu[u_idx]
        qi_m = self._mlp_Qi[i_idx]
        act  = np.concatenate([pu_m, qi_m], axis=1)   # (B, 2*mlp_factors)
        acts = [act]
        for W, b in zip(self._W, self._b):
            act = np.maximum(act @ W + b, 0.0)         # ReLU
            acts.append(act)
        mlp_out = acts[-1]                             # (B, mlp_hidden[-1])

        logit = np.concatenate([gmf_out, mlp_out], axis=1) @ self._h   # (B,)
        pred  = _sigmoid(logit)
        return pred, gmf_out, mlp_out, acts, pu, qi, pu_m, qi_m

    def _train_batch(self, u_idx: np.ndarray, i_idx: np.ndarray, y: np.ndarray) -> float:
        pred, gmf_out, mlp_out, acts, pu, qi, pu_m, qi_m = self._forward(u_idx, i_idx)
        B = len(u_idx)

        eps  = 1e-7
        loss = -np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))

        d_logit  = (pred - y) / B                                # (B,)
        concat   = np.concatenate([gmf_out, mlp_out], axis=1)
        d_h      = concat.T @ d_logit                            # (h_size,)
        d_concat = np.outer(d_logit, self._h)                    # (B, h_size)

        d_gmf_out = d_concat[:, :self.gmf_factors]
        d_mlp_out = d_concat[:, self.gmf_factors:]

        # MLP backprop through ReLU layers
        d_layer = d_mlp_out
        d_Ws, d_bs = [], []
        for k in reversed(range(len(self._W))):
            d_relu = d_layer * (acts[k + 1] > 0)   # ReLU mask
            d_Ws.insert(0, acts[k].T @ d_relu)
            d_bs.insert(0, d_relu.sum(axis=0))
            d_layer = d_relu @ self._W[k].T

        d_mlp_in = d_layer                                       # (B, 2*mlp_factors)
        d_pu_m   = d_mlp_in[:, :self.mlp_factors]
        d_qi_m   = d_mlp_in[:, self.mlp_factors:]

        # GMF gradients
        d_pu = d_gmf_out * qi
        d_qi = d_gmf_out * pu

        # Parameter updates with L2 regularisation
        lr, reg = self.lr, self.reg
        np.add.at(self._gmf_Pu, u_idx, -lr * (d_pu  + reg * pu))
        np.add.at(self._gmf_Qi, i_idx, -lr * (d_qi  + reg * qi))
        np.add.at(self._mlp_Pu, u_idx, -lr * (d_pu_m + reg * pu_m))
        np.add.at(self._mlp_Qi, i_idx, -lr * (d_qi_m + reg * qi_m))

        for k in range(len(self._W)):
            self._W[k] -= lr * (d_Ws[k] + reg * self._W[k])
            self._b[k] -= lr * d_bs[k]
        self._h -= lr * (d_h + reg * self._h)

        return float(loss)

    # ------------------------------------------------------------------ #
    def predict(self, user_id: str, product_id: str) -> float:
        u = self._user_idx.get(user_id)
        i = self._item_idx.get(product_id)
        if u is None or i is None:
            return 0.5
        pred, *_ = self._forward(np.array([u]), np.array([i]))
        return float(pred[0])

    def recommend(self, user_id: str, top_n: int = 10) -> list:
        if not self._trained:
            return []
        u = self._user_idx.get(user_id)
        if u is None:
            return []

        n_items = len(self._item_idx)
        scores, *_ = self._forward(
            np.full(n_items, u, dtype=np.int32),
            np.arange(n_items, dtype=np.int32),
        )

        pid_lookup = {row["product_id"]: row.to_dict()
                      for _, row in self._products.iterrows()}

        results = []
        for idx in np.argsort(-scores):
            pid  = self._idx_item.get(int(idx))
            if pid is None:
                continue
            sc   = float(scores[idx])
            info = pid_lookup.get(pid, {})
            results.append({
                "product_id": pid,
                "name":       info.get("name", pid),
                "category":   info.get("category", ""),
                "price":      float(info.get("price", 0)),
                "score":      round(sc, 4),
                "reason":     f"Neural CF (NeuMF): predicted interaction probability {sc:.1%}",
            })
            if len(results) >= top_n:
                break
        return results

    def get_similar_products(self, product_id: str, top_n: int = 5) -> list:
        i = self._item_idx.get(product_id)
        if i is None:
            return []
        emb   = self._mlp_Qi[i]
        norms = np.linalg.norm(self._mlp_Qi, axis=1) + 1e-8
        sims  = (self._mlp_Qi @ emb) / (norms * (np.linalg.norm(emb) + 1e-8))
        sims[i] = -np.inf

        pid_lookup = {row["product_id"]: row.to_dict()
                      for _, row in self._products.iterrows()}
        results = []
        for idx in np.argsort(-sims)[:top_n]:
            pid  = self._idx_item.get(int(idx))
            if pid is None:
                continue
            info = pid_lookup.get(pid, {})
            results.append({
                "product_id": pid,
                "name":       info.get("name", pid),
                "category":   info.get("category", ""),
                "similarity": round(float(sims[idx]), 4),
            })
        return results
