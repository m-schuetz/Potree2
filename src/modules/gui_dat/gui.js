
import * as dat from "dat.gui";
import {Potree} from "potree";

let gui = null;
let guiContent = {

	// INFOS
	"#voxels": "",
	"#points": "",
	"#nodes": "",
	"fps": "0",
	"duration(update)": "0",
	"cam.pos": "",
	"cam.target": "",
	"cam.dir": "",


	// INPUT
	"show bounding box": false,
	"use compute": false,
	"dilate": true,
	"Eye-Dome-Lighting": false,
	"High-Quality": false,
	"point budget (M)": 4,
	"point size": 3,
	"update": true,
	"debug": 0.5,

	// COLOR ADJUSTMENT
	"scalar min": 0,
	"scalar max": 2 ** 16,
	"gamma": 1,
	"brightness": 0,
	"contrast": 0,
};

// window.guiContent = guiContent;
let guiAttributes = null;
let guiScalarMin = null;
let guiScalarMax = null;

export function initGUI(potree){

	gui = new dat.GUI();
	window.gui = gui;
	window.guiContent = guiContent;
	
	{
		let stats = gui.addFolder("stats");
		stats.open();
		// stats.add(guiContent, "#voxels").listen();
		stats.add(guiContent, "#points").listen();
		stats.add(guiContent, "#nodes").listen();
		stats.add(guiContent, "fps").listen();
		stats.add(guiContent, "duration(update)").listen();
		stats.add(guiContent, "cam.pos").listen();
		stats.add(guiContent, "cam.target").listen();
		stats.add(guiContent, "cam.dir").listen();
	}

	{
		let input = gui.addFolder("input");
		input.open();

		// input.add(guiContent, "mode", [
		// 	"pixels", 
		// 	"dilate",
		// 	"HQS",
		// ]);
		// input.add(guiContent, "use compute").listen();
		input.add(guiContent, "dilate").listen();
		input.add(guiContent, "Eye-Dome-Lighting").listen();
		input.add(guiContent, "High-Quality").listen();
		input.add(guiContent, "show bounding box").listen();
		input.add(guiContent, "update").listen();
		// guiAttributes = input.add(guiContent, "attribute", ["rgba", "intensity"]).listen();
		// window.guiAttributes = guiAttributes;

		// slider
		input.add(guiContent, 'point budget (M)', 0.01, 5).listen();
		input.add(guiContent, 'point size', 1, 5);
		input.add(guiContent, 'debug', 0, 1);
	}

	// {
	// 	let input = gui.addFolder("Color Adjustments");
	// 	input.open();

	// 	guiScalarMin = input.add(guiContent, 'scalar min', 0, 2 ** 16).listen();
	// 	guiScalarMax = input.add(guiContent, 'scalar max', 0, 2 ** 16).listen();
	// 	input.add(guiContent, 'gamma', 0, 2).listen();
	// 	input.add(guiContent, 'brightness', -1, 1).listen();
	// 	input.add(guiContent, 'contrast', -1, 1).listen();
	// }


	potree.addEventListener("update", () => {

		let state = Potree.state;

		guiContent["fps"]       = state.fps.toLocaleString();
		guiContent["#voxels"]   = state.numVoxels.toLocaleString();
		guiContent["#points"]   = state.numPoints.toLocaleString();
		guiContent["#nodes"]    = state.numNodes.toLocaleString();
		guiContent["cam.pos"]   = state.camPos;
		guiContent["cam.target"]   = state.camTarget;
		guiContent["cam.dir"]   = state.camDir;
		
		// Potree.settings.mode = guiContent["mode"];
		Potree.settings.useCompute = guiContent["use compute"];
		// Potree.settings.dilateEnabled = guiContent["dilate"];
		// Potree.settings.attribute = guiContent["attribute"];
		// Potree.settings.pointBudget = guiContent["point budget (M)"] * 1_000_000;
		Potree.settings.pointSize = guiContent["point size"];
		// Potree.settings.edlEnabled = guiContent["Eye-Dome-Lighting"];
		// Potree.settings.hqsEnabled = guiContent["High-Quality"];
		// Potree.settings.updateEnabled = guiContent["update"];
		// Potree.settings.showBoundingBox = guiContent["show bounding box"];
		Potree.settings.debugU = guiContent["debug"];

	
	});

	gui.close()

}