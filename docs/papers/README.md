# Research Papers for Multi-Temporal Change Detection

This directory contains key research papers referenced in the literature review.

## Organization

Papers are organized by topic area:

```
papers/
├── README.md (this file)
├── multi_temporal_methods/      # RQ1: Multi-temporal vs bi-temporal
├── temporal_architectures/      # RQ3: LSTM, Transformers, 3D CNN
├── temporal_sampling/           # RQ2: Sampling density studies
├── benchmarks/                  # Benchmark datasets and evaluations
└── reviews/                     # Review papers and surveys
```

## Priority Reading List

### Essential (Read First)

1. **Deep Learning for SITS Review** (arXiv 2404.03936, April 2024)
   - Comprehensive review covering all three RQs
   - File: `reviews/deep_learning_sits_review_2024.pdf`
   - Link: https://arxiv.org/pdf/2404.03936

2. **ViTs for SITS** (CVPR 2023)
   - Vision transformers for satellite time series
   - File: `temporal_architectures/vits_for_sits_cvpr2023.pdf`
   - Link: https://openaccess.thecvf.com/content/CVPR2023/papers/Tarasiou_ViTs_for_SITS_Vision_Transformers_for_Satellite_Image_Time_Series_CVPR_2023_paper.pdf

3. **Change Detection Reality Check** (arXiv, Feb 2024)
   - Important benchmark cautions (WHU-CD bug)
   - File: `benchmarks/change_detection_reality_check_2024.pdf`
   - Link: https://arxiv.org/html/2402.06994

### Multi-Temporal Methods (RQ1)

4. **Enhancing change detection with LSTM** (2025)
   - LSTM-based multi-scale approach
   - File: `multi_temporal_methods/lstm_multiscale_2025.pdf`
   - Link: https://www.sciencedirect.com/science/article/abs/pii/S0273117725001814

5. **Spatial temporal fusion features** (Scientific Reports, 2025)
   - Spatial-temporal integration benefits
   - File: `multi_temporal_methods/spatial_temporal_fusion_2025.pdf`
   - Link: https://www.nature.com/articles/s41598-025-14592-x

6. **Enhanced hybrid CNN and transformer** (Scientific Reports, 2025)
   - EHCTNet for change detection
   - File: `multi_temporal_methods/ehctnet_2025.pdf`
   - Link: https://www.nature.com/articles/s41598-025-94544-7

### Temporal Architectures (RQ3)

7. **IncepTAE temporal attention** (Remote Sensing, Dec 2024)
   - Inception-enhanced temporal attention encoder
   - File: `temporal_architectures/inceptae_2024.pdf`
   - Link: https://www.mdpi.com/2072-4292/16/23/4579

8. **GL-TAE global-local features** (Remote Sensing, Jan 2023)
   - Global-local temporal attention encoder
   - File: `temporal_architectures/gl_tae_2023.pdf`
   - Link: https://www.mdpi.com/2072-4292/15/3/618

9. **U-Net enhancements for ocean remote sensing** (Journal of Remote Sensing)
   - U-TAE (U-Net with Temporal Attention Encoder)
   - File: `temporal_architectures/unet_temporal_attention.pdf`
   - Link: https://spj.science.org/doi/10.34133/remotesensing.0196

10. **Spatiotemporal fusion advances** (PMC, 2024)
    - Review of LSTM, GRU, Transformer approaches
    - File: `reviews/spatiotemporal_fusion_review_2024.pdf`
    - Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC11859923/

### Temporal Sampling Density (RQ2)

11. **Sub-annual change detection** (Remote Sensing, 2015)
    - Quarterly vs annual comparison
    - File: `temporal_sampling/subannual_change_detection_2015.pdf`
    - Link: https://www.mdpi.com/2072-4292/7/7/8705

12. **Dense time series evaluation** (Remote Sensing, 2019)
    - Comparison of BFAST, CCDC, EWMACD
    - File: `temporal_sampling/dense_timeseries_evaluation_2019.pdf`
    - Link: https://www.mdpi.com/2072-4292/11/23/2779

13. **Remote sensing time series review** (Journal of Remote Sensing)
    - Comprehensive SITS analysis review
    - File: `reviews/sits_analysis_review.pdf`
    - Link: https://spj.science.org/doi/10.34133/remotesensing.0285

### 3D CNN and Spatiotemporal Modeling

14. **Multi-scale fully convolutional network** (Taylor & Francis, 2021)
    - 3D CNN for land cover classification
    - File: `temporal_architectures/msfcn_3dcnn_2021.pdf`
    - Link: https://www.tandfonline.com/doi/full/10.1080/10095020.2021.2017237

15. **PlanetScope land cover change** (Paddy and Water Environment, 2025)
    - 3D CNN with multi-temporal PlanetScope
    - File: `multi_temporal_methods/planetscope_3dcnn_2025.pdf`
    - Link: https://link.springer.com/article/10.1007/s10333-025-01024-9

### Benchmarks and Comparisons

16. **ChangeFormer transformer-based** (PMC, 2022)
    - Transformer siamese network for change detection
    - File: `benchmarks/changeformer_2022.pdf`
    - Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC9392606/

17. **Continuous urban change detection** (arXiv, June 2024)
    - Temporal feature refinement
    - File: `temporal_sampling/continuous_urban_change_2024.pdf`
    - Link: https://arxiv.org/html/2406.17458v1

18. **Deep learning change detection review** (ScienceDirect, 2024)
    - Comprehensive review of optical change detection
    - File: `reviews/change_detection_review_2024.pdf`
    - Link: https://www.sciencedirect.com/science/article/pii/S1569843224006381

### Additional Resources

19. **Change-point detection Bayesian** (ScienceDirect, 2019)
    - Seasonal and trend analysis
    - File: `temporal_sampling/bayesian_changepoint_2019.pdf`
    - Link: https://www.sciencedirect.com/science/article/abs/pii/S0034425719301853

20. **Cross spatial temporal fusion** (arXiv, July 2025)
    - Pixel vs patch-based comparison
    - File: `temporal_architectures/cross_spatial_temporal_fusion_2025.pdf`
    - Link: https://arxiv.org/html/2507.19118v1

## How to Use

1. **For RQ1** (Multi-temporal vs bi-temporal):
   - Read papers #1, #4, #5, #6
   - Focus on LSTM benefits and temporal trajectory modeling

2. **For RQ2** (Temporal sampling density):
   - Read papers #11, #12, #13, #17
   - Focus on quarterly vs annual comparisons

3. **For RQ3** (1D vs 2D temporal modeling):
   - Read papers #1, #7, #8, #9, #10, #14, #20
   - Compare LSTM, Transformer, 3D CNN approaches

4. **For Implementation**:
   - Start with #1 (comprehensive review)
   - Check #3 for benchmark cautions
   - Use #7, #8, #9 for architecture details

## Download Instructions

All papers can be accessed through NTNU library subscriptions:

1. **Open Access** (download directly):
   - arXiv papers (just click the PDF link)
   - MDPI Remote Sensing (fully open access)
   - PMC papers (PubMed Central, open access)

2. **Via NTNU Access**:
   - ScienceDirect: https://www.sciencedirect.com (login with NTNU)
   - Springer: https://link.springer.com (automatic access on campus/VPN)
   - Taylor & Francis: https://www.tandfonline.com (NTNU subscription)

3. **Off-Campus**:
   - Use NTNU VPN
   - Or use library proxy: https://www.ntnu.no/ub

## Citation Management

Consider using BibTeX for citation management. A `references.bib` file has been created with all citations.

See: `docs/papers/references.bib`

## Notes

- PDFs are gitignored to avoid copyright issues
- Keep local copies for thesis writing
- Check license before redistributing
- Some papers are pre-prints (arXiv) - check for published versions

---

**Last Updated**: January 2026
**Total Papers**: 20 essential references
**Coverage**: All three research questions (RQ1, RQ2, RQ3)
