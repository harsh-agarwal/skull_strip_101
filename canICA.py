from nilearn.decomposition import CanICA
from nilearn.plotting import plot_prob_atlas


def perform_canICA(n_com, path_data):

    canica = CanICA(
        n_components=n_com,
        memory="nilearn_cache",
        memory_level=2,
        verbose=10,
        mask_strategy="template",
        random_state=0,
    )

    canica.fit(path_data)
    canica_components_img = canica.components_img_
    canica_components_img.to_filename("./untracked/canica_resting_state.nii.gz")
    plot_prob_atlas(canica_components_img, title="All ICA components")


if __name__ == "__main__":

    path_data = "/extern/home/harshagarwal/brain_imaging/ican_data/rsfmri_data/rs_fmri/Filtered_4DVolume.nii"
    perform_canICA(20, path_data)
