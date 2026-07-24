const vscode = require('vscode');
const fs = require('fs');

class FlagViewProvider {
  constructor(extensionUri) {
    this._extensionUri = extensionUri;
    this._view = undefined;
    this._onReset = undefined;
    this._onSetTestTime = undefined;
  }

  resolveWebviewView(webviewView) {
    this._view = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this._extensionUri, 'media')],
    };
    webviewView.webview.html = this._buildHtml(webviewView.webview);
    webviewView.webview.onDidReceiveMessage((message) => {
      if (message) {
        if (message.type === 'reset' && this._onReset) {
          this._onReset();
        }
        if (message.type === 'setTestTime' && this._onSetTestTime) {
          this._onSetTestTime(message.elapsedMinutes);
        }
      }
    });
  }

  onReset(callback) {
    this._onReset = callback;
  }

  onSetTestTime(callback) {
    this._onSetTestTime = callback;
  }

  postTick(color, elapsedMinutes) {
    if (this._view) {
      console.log(`Sending tick: color=${color}`);
      this._view.webview.postMessage({ type: 'tick', color, elapsedMinutes });
    } else {
      console.log('View not ready, tick ignored');
    }
  }

  _buildHtml(webview) {
    const mediaUri = (file) =>
      webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', file)).toString();

    const htmlPath = vscode.Uri.joinPath(this._extensionUri, 'media', 'webview.html').fsPath;
    let html = fs.readFileSync(htmlPath, 'utf8');
    const boxBoxUri = mediaUri('f1_box_box.mp3');
    console.log('BoxBox URI:', boxBoxUri);
    html = html
      .replace('{{cssUri}}', mediaUri('webview.css'))
      .replace('{{formatUri}}', mediaUri('format.js'))
      .replace('{{jsUri}}', mediaUri('webview.js'))
      .replace('{{boxBoxUri}}', boxBoxUri);
    return html;
  }
}

module.exports = { FlagViewProvider };
