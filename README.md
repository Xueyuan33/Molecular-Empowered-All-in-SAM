## Fine-grained Multi-class Nuclei Segmentation with Molecular-empowered All-in-SAM Model

## Abstract

**Purpose:** Recent developments in computational pathology have been driven by advances in Vision Foundation Models, particularly the Segment Anything Model (SAM). This model facilitates nuclei segmentation through
two primary methods: prompt-based zero-shot segmentation and the use of cell-specific SAM models for direct segmentation. These approaches enable effective segmentation across a range of nuclei and cells. However, general vision
foundation models often face challenges with fine-grained semantic segmentation, such as identifying specific nuclei subtypes or particular cells. **Approach:** In this paper, we propose the molecular-empowered All-in-SAM Model to
advance computational pathology by leveraging the capabilities of vision foundation models. This model incorporates a full-stack approach, focusing on: (1) annotation—engaging lay annotators through molecular-empowered learning
to reduce the need for detailed pixel-level annotations, (2) learning—adapting the SAM model to emphasize specific semantics, which utilizes its strong generalizability with SAM adapter, and (3) refinement—enhancing segmentation
accuracy by integrating Molecular-Oriented Corrective Learning (MOCL). **Results:** Experimental results from both in-house and public datasets show that the All-in-SAM model significantly improves cell classification performance,
even when faced with varying annotation quality. **Conclusions:** Our approach not only reduces the workload for annotators but also extends the accessibility of precise biomedical image analysis to resource-limited settings, thereby
advancing medical diagnostics and automating pathology image analysis.


## Install Segment Anything:

> pip install git+https://github.com/facebookresearch/segment-anything.git
