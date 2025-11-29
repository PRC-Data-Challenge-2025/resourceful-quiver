from resqu import FlightSet
from resqu.preprocessor import Preprocessor
import resqu.make_dataset as md

import click

@click.command
@click.option("-j", "--jobs", default=8, help="number of jobs")
@click.option(
    "-s", "--set", prompt="(train, rank, final)", help="train, rank, or final dataset"
)
def main(jobs, set):
    flight_set = FlightSet[set.upper()]    
    proc = Preprocessor(jobs=jobs, flight_set=flight_set, version='best')
    proc.run_pipe(Preprocessor.pipe_best)

if __name__=="__main__":
    main()