graph [
  node [
    id 0
    label "1 0 blue 1"
  ]
  node [
    id 1
    label "2 1 blue 2"
  ]
  node [
    id 2
    label "3 0 red 0"
  ]
  node [
    id 3
    label "4 1 blue 1"
  ]
  node [
    id 4
    label "5 1 blue 2"
  ]
  node [
    id 5
    label "0 0 red 0"
  ]
  edge [
    source 0
    target 4
    layer "L4"
    conWeight 1
    weight 4
  ]
  edge [
    source 0
    target 3
    layer "L2"
    conWeight 1
    weight 2
  ]
  edge [
    source 0
    target 5
    layer "L4"
    conWeight 1
    weight 4
  ]
  edge [
    source 1
    target 4
    layer "L5"
    conWeight 1
    weight 5
  ]
  edge [
    source 2
    target 5
    layer "L4"
    conWeight 1
    weight 4
  ]
  edge [
    source 2
    target 5
    layer "L5"
    conWeight 1
    weight 5
  ]
  edge [
    source 3
    target 4
    layer "L2"
    conWeight 1
    weight 2
  ]
  edge [
    source 4
    target 5
    layer "L2"
    conWeight 1
    weight 2
  ]
]
