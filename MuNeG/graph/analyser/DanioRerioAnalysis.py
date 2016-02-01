import graph.analyser.StatisticalAnalysis as sa

def connect_to_danio_rerio():
    sa.connect_to_data("156.17.131.228", "DanioRerio", "apopiel", "alamakota123")

if __name__ == "__main__":
    connect_to_danio_rerio()