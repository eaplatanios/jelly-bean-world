import Foundation
import Python

fileprivate let np = Python.import("numpy")
fileprivate let mpl = Python.import("matplotlib")
fileprivate let plt = Python.import("matplotlib.pyplot")
fileprivate let collections = Python.import("matplotlib.collections")
fileprivate let patches = Python.import("matplotlib.patches")

fileprivate let AGENT_RADIUS = 0.5
fileprivate let ITEM_RADIUS = 0.4
fileprivate let MAXIMUM_SCENT = 0.9

@inlinable
internal func agentPosition(_ direction: Direction) -> (x: Float, y: Float, angle: Float) {
  switch direction {
    case .up: return (x: 0.0, y: -0.1, angle: 0.0)
    case .down: return (x: 0.0, y: 0.1, angle: Float.pi)
    case .left: return (x: 0.1, y: 0.0, angle: Float.pi / 2)
    case .right: return (x: -0.1, y: 0.0, angle: 3 * Float.pi / 2)
  }
}

public final class MapVisualizer {
  let simulator: Simulator

  private var xLim: (Float, Float)
  private var yLim: (Float, Float)
  private var figId: String
  private var fig: PythonObject
  private var ax: PythonObject
  private var axAgent: PythonObject?

  public init(
    for simulator: Simulator, 
    bottomLeft: Position, 
    topRight: Position, 
    agentPerspective: Bool = true
  ) {
    self.simulator = simulator
    self.xLim = (Float(bottomLeft.x), Float(topRight.x))
    self.yLim = (Float(bottomLeft.y), Float(topRight.y))
    self.figId = "MapVisualizer Figure 1"
    plt.ion()
    if agentPerspective {
      let figure = plt.subplots(nrows: 1, ncols: 2, num: self.figId)
      self.fig = figure[0]
      self.ax = figure[1][0]
      self.axAgent = figure[1][1]
      self.fig.set_size_inches(w: 18, h: 9)
    } else {
      let figure = plt.subplots(num: self.figId)
      self.fig = figure[0]
      self.ax = figure[1]
      self.axAgent = nil
      self.fig.set_size_inches(w: 9, h: 9)
    }
    self.fig.tight_layout()
  }

  deinit {
    plt.close(self.fig)
  }

  private func pause(_ interval: Float) {
    let backend = plt.rcParams["backend"]
    if Array(mpl.rcsetup.interactive_bk).contains(backend) {
      let figManager = mpl._pylab_helpers.Gcf.get_active()
      if figManager != Python.None {
        let canvas = figManager.canvas
        if Bool(canvas.figure.stale)! {
          canvas.draw()
        }
        canvas.start_event_loop(interval)
      }
    }
  }

  public func draw() {
    let bottomLeft = Position(x: Int64(floor(self.xLim.0)), y: Int64(floor(self.yLim.0)))
    let topRight = Position(x: Int64(ceil(self.xLim.1)), y: Int64(ceil(self.yLim.1)))
    let map = self.simulator.map(bottomLeft: bottomLeft, topRight: topRight)
    let n = Int(self.simulator.config.patchSize)
    self.ax.clear()
    self.ax.set_xlim(self.xLim.0, self.xLim.1)
    self.ax.set_ylim(self.yLim.0, self.yLim.1)

    // Draw all the map patches.
    for patch in map.patches {
      let color = PythonObject(
        tupleContentsOf: patch.fixed ? [0.0, 0.0, 0.0, 0.3] : [0.0, 0.0, 0.0, 0.1])
      let a = Python.slice(Python.None, Python.None, Python.None)
      let x = Int(patch.position.x)
      let y = Int(patch.position.y)

      let verticalLines = np.empty(shape: [n + 1, 2, 2])
      verticalLines[a, 0, 0] = np.add(np.arange(n + 1), x * n) - 0.5
      verticalLines[a, 0, 1] = np.subtract(y * n, 0.5)
      verticalLines[a, 1, 0] = np.add(np.arange(n + 1), x * n) - 0.5
      verticalLines[a, 1, 1] = np.add(y * n, n) - 0.5
      let verticalLineCol = collections.LineCollection(
        segments: verticalLines, 
        colors: color, 
        linewidths: 0.4, 
        linestyle: "solid")
      self.ax.add_collection(verticalLineCol)

      let horizontalLines = np.empty(shape: [n + 1, 2, 2])
      horizontalLines[a, 0, 0] = np.subtract(x * n, 0.5)
      horizontalLines[a, 0, 1] = np.add(np.arange(n + 1), y * n) - 0.5
      horizontalLines[a, 1, 0] = np.add(x * n, n) - 0.5
      horizontalLines[a, 1, 1] = np.add(np.arange(n + 1), y * n) - 0.5
      let horizontalLinesCol = collections.LineCollection(
        segments: horizontalLines, 
        colors: color, 
        linewidths: 0.4, 
        linestyle: "solid")
      self.ax.add_collection(horizontalLinesCol)

      var agentItemPatches = [PythonObject]()

      // Add the agent patches.
      for agent in patch.agents {
        let x = Float(agent.position.x)
        let y = Float(agent.position.y)
        let position = agentPosition(agent.direction)
        agentItemPatches.append(patches.RegularPolygon(
          [x + position.x, y + position.y],
          numVertices: 3,
          radius: AGENT_RADIUS,
          orientation: position.angle,
          facecolor: self.simulator.config.agentColor.scalars,
          edgecolor: [0.0, 0.0, 0.0],
          linestyle: "solid",
          linewidth: 0.4))
      }

      // Add the item patches.
      for item in patch.items {
        let id = item.itemType
        let x = Float(item.position.x)
        let y = Float(item.position.y)
        if self.simulator.config.items[id].blocksMovement {
          agentItemPatches.append(patches.Rectangle(
            [x - 0.5, y - 0.5],
            width: 1.0,
            height: 1.0,
            facecolor: self.simulator.config.items[id].color.scalars,
            edgecolor: [0.0, 0.0, 0.0],
            linestyle: "solid",
            linewidth: 0.4))
        } else {
          agentItemPatches.append(patches.Circle(
            [x, y],
            radius: ITEM_RADIUS,
            facecolor: self.simulator.config.items[id].color.scalars,
            edgecolor: [0.0, 0.0, 0.0],
            linestyle: "solid",
            linewidth: 0.4))
        }
      }

      // Convert 'scent' to a numpy array and transform it into 
      // a subtractive color space (so that zero corresponds to white).
      var scentImg = patch.scent.makeNumpyArray()
      scentImg = np.divide(np.log(np.power(scentImg, 0.4) + 1), MAXIMUM_SCENT)
			scentImg = np.clip(scentImg, 0.0, 1.0)
			scentImg = 1.0 - 0.5 * np.dot(scentImg, [[0, 1, 1], [1, 0, 1], [1, 1, 0]])
      let left = Float(x * n) - 0.5
      let right = Float(x * n + n) - 0.5
      let bottom = Float(y * n) - 0.5
      let top = Float(y * n + n) - 0.5
			self.ax.imshow(np.rot90(scentImg), extent: [left, right, bottom, top])
      
      // Add the agent and item patches to the plots.
      let agentItemPatchCol = collections.PatchCollection(agentItemPatches, match_original: true)
      self.ax.add_collection(agentItemPatchCol)
    }

    // Draw the agent's perspective.
    if self.simulator.agents.keys.contains(0) {
      if let axAgent = self.axAgent {
        let agent = self.simulator.agents[0]!
        let r = Int(self.simulator.config.visionRange)
        let a = Python.slice(Python.None, Python.None, Python.None)
        
        axAgent.clear()
        axAgent.set_xlim(-Float(r) - 0.5, Float(r) + 0.5)
        axAgent.set_ylim(-Float(r) - 0.5, Float(r) + 0.5)

        let verticalLines = np.empty(shape: [2 * r, 2, 2])
        verticalLines[a, 0, 0] = np.subtract(np.arange(2 * r), r) + 0.5
        verticalLines[a, 0, 1] = np.subtract(-r, 0.5)
        verticalLines[a, 1, 0] = np.subtract(np.arange(2 * r), r) + 0.5
        verticalLines[a, 1, 1] = np.add(r, 0.5)
        let verticalLineCol = collections.LineCollection(
          segments: verticalLines, 
          colors: [0.0, 0.0, 0.0, 0.3],
          linewidths: 0.4, 
          linestyle: "solid")
        axAgent.add_collection(verticalLineCol)

        let horizontalLines = np.empty(shape: [2 * r, 2, 2])
        horizontalLines[a, 0, 0] = np.subtract(-r, 0.5)
        horizontalLines[a, 0, 1] = np.subtract(np.arange(2 * r), r) + 0.5
        horizontalLines[a, 1, 0] = np.add(r, 0.5)
        horizontalLines[a, 1, 1] = np.subtract(np.arange(2 * r), r) + 0.5
        let horizontalLinesCol = collections.LineCollection(
          segments: horizontalLines, 
          colors: [0.0, 0.0, 0.0, 0.3],
          linewidths: 0.4, 
          linestyle: "solid")
        axAgent.add_collection(horizontalLinesCol)

        // Add the agent patch to the plot.
        let position = agentPosition(.up)
        let agentPatch = patches.RegularPolygon(
          [position.x, position.y],
          numVertices: 3,
          radius: AGENT_RADIUS,
          orientation: position.angle,
          facecolor: self.simulator.config.agentColor.scalars,
          edgecolor: [0.0, 0.0, 0.0],
          linestyle: "solid",
          linewidth: 0.4)
        let agentPatchCol = collections.PatchCollection([agentPatch], match_original: true)
        axAgent.add_collection(agentPatchCol)

        // Convert 'vision' to a numpy array and transform it into 
        // a subtractive color space (so that zero corresponds to white).
        var visionImg = agent.vision!.makeNumpyArray()
        visionImg = 1.0 - 0.5 * np.dot(visionImg, [[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        let left = -Float(r) - 0.5
        let right = Float(r) + 0.5
        let bottom = -Float(r) - 0.5
        let top = Float(r) + 0.5
			  axAgent.imshow(np.rot90(visionImg), extent: [left, right, bottom, top])
      }
    }

    self.pause(1e-16)
    plt.draw()

    let pltXLim = self.ax.get_xlim()
    let pltYLim = self.ax.get_ylim()
    self.xLim = (Float(pltXLim[0])!, Float(pltXLim[1])!)
    self.yLim = (Float(pltYLim[0])!, Float(pltYLim[1])!)
  }
}
