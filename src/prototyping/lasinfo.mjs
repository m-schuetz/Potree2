
import * as fs from "fs";

let files = [
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120C7101A_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120C7101C_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120C7101B_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120C7101D_1.laz",

	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120D7322D_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120D7322B_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120D7322A_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120D7322C_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120D7323A_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120D7323C_1.laz",

	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120C7102A_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120C7102C_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120C7102B_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120C7102D_1.laz",

	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120C7103A_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120C7103C_1.laz",
	
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120D7321D_1.laz",
	"E:/dev/pointclouds/opentopography/CA13_SAN_SIM/ot_35120D7321B_1.laz",
	
];

let sum = 0;
for(let file of files){
	let buffer = fs.readFileSync(file);

	let numPoints = buffer.readUInt32LE(107);
	sum += numPoints;
}

console.log(sum.toLocaleString());