from f1_core.data import FEATURES, build_training_data, feature_matrix


def test_training_data_shape_matches_script():
    data = build_training_data()
    assert len(data) == 330  # 15 races x 22 drivers, as printed by sepang/model.py
    assert set(FEATURES) <= set(data.columns)
    assert int(data["target"].sum()) == 45  # 3 podium rows x 15 races


def test_feature_matrix_has_no_nans():
    X, y = feature_matrix(build_training_data())
    assert X.shape == (330, 22)
    assert list(X.columns) == FEATURES
    assert not X.isna().any().any()
    assert set(y.unique()) == {0, 1}
