# Keep notes about what we're doing here.

Models:
https://github.com/fizyr/tf-retinanet
* Make a docker image that can run this
* Make a sample training set that can train this
* Make a sample inference set that can detect this

Data:
gcloud
project natgeo-dams
data: natgeo-dams-data bucket
    * ecoshards:
    * status database: this database has a `bounding_box_to_mosaic` table that has `bounding_box`, `quad_id` to map to the known-dam-quads in the pattern 4ce5863a-fb3f-4cad-a899-b8c053af1858_[`quad_id`].tif
    * known-dam-quads: full quads from Planet that have known dams in them
    * not-a-dam-imagery: pngs made by Lisa and Rich for data that do not have dams in them.


***
(old-- don't use anymore because MS is confusing) ecoshard storage account:
    resource-group: natgeo_dams_phase_2
    account name: ecoshards
    /subscriptions/4e1c44aa-f152-4aeb-aca0-8f01ffc8fb56/resourceGroups/natgeo_dams_phase_2/providers/Microsoft.Storage/storageAccounts/ecoshards
    /subscriptions/4e1c44aa-f152-4aeb-aca0-8f01ffc8fb56/resourceGroups/natgeo_dams_phase_2/providers/Microsoft.Resources/deployments/Microsoft.StorageAccount-20200304140219/operations/A18A3ED09A02CCDC