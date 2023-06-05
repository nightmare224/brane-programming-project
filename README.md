# brane-programming-project
The assignment is about using the Brane framework to implement a data processing pipeline for the assignment of Web services and Cloud-Based Systems course in UvA.

We aim to find the key factors of heart disease through the dataset from [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease). We implemented the compute pipeline to do data preprocessing, model training, and visualization as well as the report pipeline to gather the result of the compute pipeline into the HTML report. Through the report, we can find out the key factors of heart disease.

<img src="https://github.com/nightmare224/brane-programming-project/blob/master/docs/images/pipeline_overview.png" alt="pipeline_overview"/>

# Getting Started
## Build
To build the required packages and dataset in the Brane environment, run:
```bash
bash brane-programming-project/brane-heart-disease/build.sh
```

If the packages and dataset build successfully, you should see:

<img src="https://github.com/nightmare224/brane-programming-project/blob/master/docs/images/build.png" alt="build"/>

## Run

There are two pipeline in total. The first one is compute pipeline. The second one is report pipeline. All the computation are in the compute pipeline. The report pipeline is used to collect all the figures that generate from compute pipeline into single HTML report.

To trigger compute pipeline, run:

```bash
bash brane-programming-project/brane-heart-disease/scripts/pipeline.sh
```

If the compute pipeline run sucessfully, you should see:

<img src="https://github.com/nightmare224/brane-programming-project/blob/master/docs/images/compute_pipeline.png" alt="compute_pipeline"/>

To trigger report pipeline, run:

```bash
bash brane-programming-project/brane-heart-disease/scripts/report.sh
```

If the report pipeline run sucessfully, you should see:

<img src="https://github.com/nightmare224/brane-programming-project/blob/master/docs/images/report_pipeline.png" alt="report_pipeline"/>

## Usage

If the report pipeline run successfully,  you can find out the filepath of final report by running:

```bash
brane data path heart-disease-report
```

This command would return the path of directory as shown in the below figure.

<img src="https://github.com/nightmare224/brane-programming-project/blob/master/docs/images/report_directory.png" alt="report_pipeline"/>

The directory should contain several HTML files. Download the whole directory and then open the `report.html`. You would see the menu like this:

<img src="https://github.com/nightmare224/brane-programming-project/blob/master/docs/images/report_menu.png" alt="report_menu"/>

This report would contain all the figures that genreated in the compute pipeline. To see the demo report, check [here](https://github.com/nightmare224/brane-programming-project/blob/master/docs/report/).



# Discussion

As you can see, we have two pipeline: compute pipeline and report pipeline. In fact, there is no reason that we split it into two pipeline. It should be in the single pipeline. The reason of having two pipeline is because Brane seems cannot use the commit dataset instantly in same package in same pipeline. 

For example:

```bs
import visualization;

let fig := feature_importance();
commit_result("heart-disease-report", fig);
// will get heart-disease-report not found
let data := new Data { name := "heart-disease-report" };
let report := generate_report(fig);
```

Although the workaroud is to pass *IntermediateType* data, it cannot obtain the output file as a single directory accross multiple function.

```bs
import visualization;

let fig := feature_importance();
fig := model_report();
fig := heart_disease_positive_ratio();
fig := heart_disease_positive_ratio();
fig := heart_disease_positive_ratio();
fig := heart_disease_positive_ratio();
fig := heart_disease_positive_ratio();

let report := generate_report(fig);
```



# References

## Data Analysis

- https://www.kaggle.com/code/jaewook704/heart-disease-scoring-who-is-dangerous

  >Refer to the description of features and the positive ratio plot method.

- https://www.kaggle.com/code/jayrdixit/heart-disease-indicators

  >Refer to the way it get feature importance.

## Brane

- https://github.com/marinoandrea/disaster-tweets-brane

  >Refer to the Brane package and Brane script, and also the project layout.

## Visualization

- https://plotly.com/python/table/

  >Refer to the dataset showing in table.

- https://codepen.io/GoostCreative/pen/jOawZbZ

  >The template of report.
