//! Retain reusable point buffers only when their spare capacity is modest.

pub(super) fn trim_spare_capacity(points: &mut Vec<[f64; 3]>) {
    if points.is_empty() {
        *points = Vec::new();
    } else if points.capacity() > points.len().saturating_mul(2) {
        points.shrink_to_fit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_sparse_buffers_release_storage() {
        let mut points = Vec::with_capacity(100_000);
        points.push([1.0, 2.0, 3.0]);
        trim_spare_capacity(&mut points);
        assert_eq!(points, vec![[1.0, 2.0, 3.0]]);
        assert_eq!(points.capacity(), 1);
        points.clear();
        trim_spare_capacity(&mut points);
        assert_eq!(points.capacity(), 0);
    }

    #[test]
    fn dense_buffers_keep_their_allocation() {
        let mut points = Vec::with_capacity(8);
        points.extend([[1.0; 3]; 4]);
        let pointer = points.as_ptr();
        trim_spare_capacity(&mut points);
        assert_eq!(points.as_ptr(), pointer);
        assert_eq!(points.capacity(), 8);
    }
}
