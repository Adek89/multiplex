import graph.analyser.StatisticalAnalysis as sa

def connect_to_danio_rerio():
    return sa.connect_to_data("156.17.131.228", "DanioRerio", "apopiel", "alamakota123")

if __name__ == "__main__":
    cursor = connect_to_danio_rerio()
    cursor.execute('select rwc_iter from DanioRerio.daniorerio.results')
    results = cursor.fetchall()
    data = sorted(results.ado_results[0])
    sa.draw_histogram(data)