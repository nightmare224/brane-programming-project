# brane-programming-project
The assignment is about using the Brane framework1 to implement a data processing pipeline for the assignment of Web services and Cloud-Based Systems course in UvA.





# Getting Started
## Build
To build the required packages and dataset in the BRANE environment, run:
```bash
bash brane-programming-project/brane-heart-disease/build.sh
```

If the packages and dataset build successfully, you should see:

<img src="https://github.com/nightmare224/brane-programming-project/blob/master/docs/images/build.png" alt="build"/>

## Pipeline

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
