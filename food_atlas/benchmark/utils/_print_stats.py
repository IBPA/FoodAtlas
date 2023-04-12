def print_statistics(
        data_cocoa_mine,
        data_garlic_mine,
        data_cocoa_atals,
        data_garlic_atals,
        ):
    cids_cocoa_mine = set(data_cocoa_mine['pubchem_id'].tolist())
    cids_cocoa_atals = set(data_cocoa_atals['pubchem_id'].tolist())
    print(f"Food: cocoa")
    print(f"# of unique PubChem IDs in FM     : {len(cids_cocoa_mine)}")
    print(f"# of unique PubChem IDs in FA     : {len(cids_cocoa_atals)}")
    print(
        f"# of unique PubChem IDs overlapped: "
        f"{len(cids_cocoa_mine & cids_cocoa_atals)}"
    )
    print()

    cids_garlic_mine = set(data_garlic_mine['pubchem_id'].tolist())
    cids_garlic_atals = set(data_garlic_atals['pubchem_id'].tolist())
    print(f"Food: garlic")
    print(f"# of unique PubChem IDs in FM     : {len(cids_garlic_mine)}")
    print(f"# of unique PubChem IDs in FA     : {len(cids_garlic_atals)}")
    print(
        f"# of unique PubChem IDs overlapped: "
        f"{len(cids_garlic_mine & cids_garlic_atals)}"
    )
    print()
