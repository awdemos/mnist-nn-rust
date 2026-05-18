use csv::Reader;
use ndarray::{Array1, Array2};
use std::fs::File;

pub fn load_csv(path: &str) -> (Array2<f64>, Array1<usize>) {
    let file = File::open(path).unwrap();
    let mut rdr = Reader::from_reader(file);
    let mut images = vec![];
    let mut labels = vec![];

    for result in rdr.records() {
        let record = result.unwrap();
        let label: usize = record[0].parse().unwrap();
        let pixels: Vec<f64> = record.iter().skip(1)
            .map(|s| s.parse::<f64>().unwrap() / 255.0)
            .collect();
        images.push(pixels);
        labels.push(label);
    }

    let n = images.len();
    let flat: Vec<f64> = images.into_iter().flatten().collect();
    let x = Array2::from_shape_vec((n, 784), flat).unwrap();
    let y = Array1::from_shape_vec(n, labels).unwrap();
    (x, y)
}
