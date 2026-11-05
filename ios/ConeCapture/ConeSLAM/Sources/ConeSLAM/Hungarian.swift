/// Rectangular minimum-cost assignment (scipy `linear_sum_assignment`): every row of the smaller side
/// is assigned. `cost` is row-major `rows x cols`. Returns (row, col) pairs sorted by row.
func linearSumAssignment(_ cost: [Double], rows n: Int, cols m: Int) -> [(Int, Int)] {
    guard n > 0, m > 0 else { return [] }
    if n > m {
        var t = [Double](repeating: 0, count: n * m)
        for i in 0..<n { for j in 0..<m { t[j * n + i] = cost[i * m + j] } }
        return linearSumAssignment(t, rows: m, cols: n).map { ($0.1, $0.0) }.sorted { $0.0 < $1.0 }
    }
    // Shortest augmenting path with potentials (1-indexed, n <= m).
    let inf = Double.infinity
    var u = [Double](repeating: 0, count: n + 1), v = [Double](repeating: 0, count: m + 1)
    var p = [Int](repeating: 0, count: m + 1), way = [Int](repeating: 0, count: m + 1)
    for i in 1...n {
        p[0] = i
        var j0 = 0
        var minv = [Double](repeating: inf, count: m + 1)
        var used = [Bool](repeating: false, count: m + 1)
        repeat {
            used[j0] = true
            let i0 = p[j0]
            var delta = inf, j1 = 0
            for j in 1...m where !used[j] {
                let current = cost[(i0 - 1) * m + (j - 1)] - u[i0] - v[j]
                if current < minv[j] { minv[j] = current; way[j] = j0 }
                if minv[j] < delta { delta = minv[j]; j1 = j }
            }
            for j in 0...m {
                if used[j] { u[p[j]] += delta; v[j] -= delta } else { minv[j] -= delta }
            }
            j0 = j1
        } while p[j0] != 0
        repeat {
            let j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
        } while j0 != 0
    }
    var out: [(Int, Int)] = []
    for j in 1...m where p[j] != 0 { out.append((p[j] - 1, j - 1)) }
    return out.sorted { $0.0 < $1.0 }
}
