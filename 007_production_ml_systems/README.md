# Notes from Lectures, Labs and Readings

### ML Model Code counts for roughly 5% all code
<img src="./pictures/ml_code_minority.png" alt="drawing" width="600"/>

### Other components in ML Production System
<img src="./pictures/other_components_in_ml_system.png" alt="drawing" width="600"/>

### Reuse other pieces of code
<img src="./pictures/reuse_generic_pieces.png" alt="drawing" width="600"/>

### Choosing where your data should be stored
<img src="./pictures/data_storage_options.png" alt="drawing" width="600"/>

### ETL Push Solution Architecture
- This architecture is best for those wanting ad hoc or invent based loading
<img src="./pictures/etl_push_solution_architecture.png" alt="drawing" width="400"/>

### ETL Pull Solution Architecture
- Pull models are better for when there is a repeatable process and scheduled interval, instead of firing on-demand
<img src="./pictures/etl_pull_solution_architecture.png" alt="drawing" width="400"/>

The architecture for the pull model is very similar to the push model with the only difference being how the entire pipeline is invoked (between Cloud Composer and Cloud Functions). This pattern expects that the file will be waiting there to set schedule instead of starting upon an event. As you could have guessed, pull models are better for when there is a repeatable process and scheduled interval, instead of firing on-demand. 