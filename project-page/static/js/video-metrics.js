'use strict';

const videoMetrics = {
  "spatial": {
    "ppo-hopper": [
      {
        "label": "CAPS spatial-only",
        "reward": 1072.8888979202375,
        "smoothness": 0.202175276046612
      },
      {
        "label": "LipsNet++",
        "reward": 970.0133332240835,
        "smoothness": 0.24152287173268694
      },
      {
        "label": "CATS",
        "reward": 3295.7818695745987,
        "smoothness": 0.22201309840281125
      }
    ],
    "ppo-walker": [
      {
        "label": "CAPS spatial-only",
        "reward": 2962.245244364567,
        "smoothness": 0.4081484005775355
      },
      {
        "label": "LipsNet++",
        "reward": 2514.336440525009,
        "smoothness": 0.3508462604604888
      },
      {
        "label": "CATS",
        "reward": 4047.6150574774297,
        "smoothness": 0.3872597130168798
      }
    ],
    "sac-hopper": [
      {
        "label": "CAPS spatial-only",
        "reward": 665.8644158922227,
        "smoothness": 0.21783773516896918
      },
      {
        "label": "LipsNet++",
        "reward": 1043.9210647356144,
        "smoothness": 0.2582770505419632
      },
      {
        "label": "CATS",
        "reward": 3461.793724253985,
        "smoothness": 0.24203621730321487
      }
    ],
    "sac-walker": [
      {
        "label": "CAPS spatial-only",
        "reward": 642.2419802709455,
        "smoothness": 0.4604078596933685
      },
      {
        "label": "LipsNet++",
        "reward": 1790.172932935782,
        "smoothness": 0.4847219732846259
      },
      {
        "label": "CATS",
        "reward": 4627.476076914947,
        "smoothness": 0.45318052668939585
      }
    ]
  },
  "noise": {
    "0.0": [
      {
        "label": "PPO",
        "reward": 2668.072294354439,
        "smoothness": 0.9490583211873354
      },
      {
        "label": "CATS",
        "reward": 2941.702742576599,
        "smoothness": 0.23962280695439364
      }
    ],
    "0.1": [
      {
        "label": "PPO",
        "reward": 2596.8968781232834,
        "smoothness": 1.40510655019919
      },
      {
        "label": "CATS",
        "reward": 2819.410327196121,
        "smoothness": 0.29813161187191256
      }
    ],
    "0.2": [
      {
        "label": "PPO",
        "reward": 2301.5112931728363,
        "smoothness": 1.6814968245093744
      },
      {
        "label": "CATS",
        "reward": 2418.2848967313766,
        "smoothness": 0.3700463350931578
      }
    ],
    "0.3": [
      {
        "label": "PPO",
        "reward": 1784.2931687831879,
        "smoothness": 1.8383874660324289
      },
      {
        "label": "CATS",
        "reward": 2263.350906610489,
        "smoothness": 0.5134210987068198
      }
    ],
    "0.4": [
      {
        "label": "PPO",
        "reward": 1598.1598232984543,
        "smoothness": 1.8392522176057138
      },
      {
        "label": "CATS",
        "reward": 1977.240792632103,
        "smoothness": 0.576122241091192
      }
    ],
    "0.5": [
      {
        "label": "PPO",
        "reward": 1346.698276758194,
        "smoothness": 1.8591887977706074
      },
      {
        "label": "CATS",
        "reward": 1715.9482100009918,
        "smoothness": 0.6295292796303142
      }
    ]
  },
  "real": [
    {
      "label": "PPO",
      "success": 0.0,
      "success5": 0.0,
      "successes5": 0,
      "successes": 0,
      "targets": 6,
      "smoothness": 5.307446844738706
    },
    {
      "label": "CATS",
      "success": 83.33333333333333,
      "success5": 16.666666666666668,
      "successes5": 1,
      "successes": 5,
      "targets": 6,
      "smoothness": 0.09721585651099486
    }
  ]
};
