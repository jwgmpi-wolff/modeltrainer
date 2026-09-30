using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: AssemblyTitle("Local Model Lab Setup")]
[assembly: AssemblyProduct("Local Model Lab")]
[assembly: AssemblyCompany("Jerry Wolff")]
[assembly: AssemblyVersion("1.0.1.0")]
[assembly: AssemblyFileVersion("1.0.1.0")]

internal static class Installer
{
    private const string AppName = "Local Model Lab";
    private const string Version = "1.0.1";
    private const string UninstallKey = @"Software\Microsoft\Windows\CurrentVersion\Uninstall\LocalModelLab";

    [STAThread]
    private static int Main(string[] args)
    {
        bool silent = Array.Exists(args, value => string.Equals(value, "/S", StringComparison.OrdinalIgnoreCase));
        string installDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Programs",
            AppName);

        try
        {
            if (!silent)
            {
                DialogResult result = MessageBox.Show(
                    "Install Local Model Lab for the current Windows user?",
                    "Local Model Lab Setup",
                    MessageBoxButtons.OKCancel,
                    MessageBoxIcon.Information);
                if (result != DialogResult.OK) return 1;
            }

            StopInstalledProcesses(installDirectory);
            if (Directory.Exists(installDirectory)) Directory.Delete(installDirectory, true);
            Directory.CreateDirectory(installDirectory);

            using (Stream payload = Assembly.GetExecutingAssembly().GetManifestResourceStream("ModelPayload.zip"))
            {
                if (payload == null) throw new InvalidOperationException("The application payload is missing.");
                string temporaryZip = Path.Combine(Path.GetTempPath(), "local-model-lab-" + Guid.NewGuid() + ".zip");
                try
                {
                    using (FileStream output = File.Create(temporaryZip)) payload.CopyTo(output);
                    ZipFile.ExtractToDirectory(temporaryZip, installDirectory);
                }
                finally
                {
                    if (File.Exists(temporaryZip)) File.Delete(temporaryZip);
                }
            }

            string executablePath = Path.Combine(installDirectory, AppName + ".exe");
            string uninstallerPath = Path.Combine(installDirectory, "Uninstall " + AppName + ".exe");
            if (!File.Exists(executablePath)) throw new InvalidOperationException("The installed application executable is missing.");
            if (!File.Exists(uninstallerPath)) throw new InvalidOperationException("The installed uninstaller is missing.");

            CreateShortcuts(executablePath, installDirectory);
            RegisterApplication(executablePath, uninstallerPath, installDirectory);

            if (!silent)
            {
                MessageBox.Show(
                    "Local Model Lab was installed successfully.",
                    "Local Model Lab Setup",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information);
            }

            Process.Start(new ProcessStartInfo(executablePath) {
                WorkingDirectory = installDirectory,
                UseShellExecute = true
            });
            return 0;
        }
        catch (Exception error)
        {
            MessageBox.Show(
                error.Message,
                "Local Model Lab installation failed",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return 2;
        }
    }

    private static void StopInstalledProcesses(string installDirectory)
    {
        string normalizedRoot = Path.GetFullPath(installDirectory).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        foreach (Process process in Process.GetProcesses())
        {
            try
            {
                string processPath = process.MainModule == null ? null : process.MainModule.FileName;
                if (processPath != null && Path.GetFullPath(processPath).StartsWith(normalizedRoot, StringComparison.OrdinalIgnoreCase))
                {
                    process.Kill();
                    process.WaitForExit(10000);
                }
            }
            catch (System.ComponentModel.Win32Exception) { }
            catch (InvalidOperationException) { }
        }
    }

    private static void CreateShortcuts(string executablePath, string installDirectory)
    {
        string desktopShortcut = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory),
            AppName + ".lnk");
        string startMenuShortcut = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Programs),
            AppName + ".lnk");

        foreach (string shortcutPath in new[] { desktopShortcut, startMenuShortcut })
        {
            Type shellType = Type.GetTypeFromProgID("WScript.Shell");
            dynamic shell = Activator.CreateInstance(shellType);
            dynamic shortcut = shell.CreateShortcut(shortcutPath);
            shortcut.TargetPath = executablePath;
            shortcut.WorkingDirectory = installDirectory;
            shortcut.IconLocation = executablePath + ",0";
            shortcut.Description = "Launch Local Model Lab";
            shortcut.Save();
        }
    }

    private static void RegisterApplication(string executablePath, string uninstallerPath, string installDirectory)
    {
        using (RegistryKey key = Registry.CurrentUser.CreateSubKey(UninstallKey))
        {
            key.SetValue("DisplayName", AppName);
            key.SetValue("DisplayVersion", Version);
            key.SetValue("DisplayIcon", executablePath + ",0");
            key.SetValue("Publisher", "Jerry Wolff");
            key.SetValue("InstallLocation", installDirectory);
            key.SetValue("UninstallString", "\"" + uninstallerPath + "\"");
            key.SetValue("NoModify", 1, RegistryValueKind.DWord);
            key.SetValue("NoRepair", 1, RegistryValueKind.DWord);
        }
    }
}
