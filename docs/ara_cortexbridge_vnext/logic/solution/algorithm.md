# Algorithm

## Implemented Cross-Geometry Objective

Given projected normalized surface and ribbon tokens:

\[
S=\operatorname{norm}(W_sS_0),\qquad
R=\operatorname{norm}(W_rR_0),
\]

form the subject-local similarity matrix:

\[
M_{b,i,j}=S_{b,i}^{\top}R_{b,j}/\tau.
\]

For ROI-index target \(y_i=i\):

\[
L_{CG}=\frac{1}{2}
\left[
CE(M,y)+CE(M^{\top},y)
\right].
\]

The implemented Teacher objective is:

\[
L_T=L_{CE}+\lambda_{clip}L_{clip}+\lambda_{proto}L_{proto}
+\lambda_{rel}L_{rel}+\lambda_{CG}L_{CG}.
\]

## Implemented Subject-Specific Audit

Let the four population means be:

\[
\mu_{ss}=E[S_{b,i}^{\top}R_{b,i}],\quad
\mu_{ds}=E[S_{b,i}^{\top}R_{b',i}],
\]

\[
\mu_{sd}=E[S_{b,i}^{\top}R_{b,j}],\quad
\mu_{dd}=E[S_{b,i}^{\top}R_{b',j}].
\]

Then:

\[
G_{subject}=\mu_{ss}-\mu_{ds},\qquad
G_{ROI}=\mu_{ds}-\mu_{dd}.
\]

The descriptive shares are each component divided by \(\mu_{ss}-\mu_{dd}\).

## Planned Graph Algorithm

For base tokens \(Z\):

\[
E=QK^\top/\sqrt d+\beta_B\phi(A_{boundary})+\beta_HA_{hom}.
\]

No remote edge is hard-masked. Then:

\[
A_T=\operatorname{softmax}_j(E),\qquad
\Delta Z=W_o(A_TV),
\]

\[
Z_G=Z+\sigma(g)\odot\Delta Z,
\]

followed by Pre-LN FFN residual processing.

### Pseudocode

```python
def teacher_forward(batch, topology):
    deep, multiscale, global_mri = mri_encoder(batch.image)
    surface = surface_pool(surface_encoder(batch.surface), batch.surface_masks)
    volume = legacy_volume_pool(deep, batch.local_masks, batch.roi_masks)

    ribbon = ribbon_sampler(multiscale[0], batch.ribbon_grid, batch.vertex_roi)
    loss_cg = hard_bidirectional_roi_loss(surface, ribbon)

    z_surface, z_volume = roi_clip(surface, volume)
    z_base = teacher_fusion(z_surface, z_volume)

    if graph_enabled:
        z_graph, graph = roi_graph(z_base, topology)
    else:
        z_graph, graph = z_base, None

    roi_logits = roi_classifier(z_graph)
    logits = global_classifier(global_mri) + roi_gate * roi_logits
    prototypes = prototype_bank(z_graph)
    return logits, z_graph, graph, prototypes, loss_cg
```

## Planned Student Distillation

\[
L_S=L_{CE}+\alpha L_{logit}+\beta L_{ROI}+\gamma L_{graphKD}+\delta L_{protoKD}.
\]

ROI-state loss aligns `z_teacher_graph` and `z_student_graph`; graph KD applies row-wise KL; prototype KD shares a frozen Teacher prototype reference in the first implementation.

## Complexity

- Ribbon sampling: \(O(BNLC)\) sampled feature work plus scatter pooling.
- Subject-local correspondence: \(O(BK^2D)\), with small fixed \(K\).
- Dense ROI graph attention: \(O(BHK^2D_h)\); feasible for the fixed DKT ROI count.
- Student MRI query attention: \(O(BKVD)\), where \(V\) is flattened feature-map size.

