using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: AssemblyTitle("Uninstall Local Model Lab")]
[assembly: AssemblyProduct("Local Model Lab")]
[assembly: AssemblyCompany("Jerry Wolff")]
[assembly: AssemblyVersion("1.1.0.0")]
[assembly: AssemblyFileVersion("1.1.0.0")]

internal static class Uninstaller
{
    private const string AppName = "Local Model Lab";
    private const string UninstallKey = @"Software\Microsoft\Windows\CurrentVersion\Uninstall\LocalModelLab";

    [STAThread]
    private static int Main(string[] args)
    {
        bool silent = Array.Exists(args, value => string.Equals(value, "/S", StringComparison.OrdinalIgnoreCase));
        if (!silent)
        {
            DialogResult result = MessageBox.Show(
                "Remove Local Model Lab from this computer?",
                "Uninstall Local Model Lab",
                MessageBoxButtons.OKCancel,
                MessageBoxIcon.Warning);
            if (result != DialogResult.OK) return 1;
        }

        string installDirectory = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
        StopInstalledProcesses(installDirectory);
        DeleteShortcut(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), AppName + ".lnk"));
        DeleteShortcut(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), AppName + ".lnk"));
        Registry.CurrentUser.DeleteSubKeyTree(UninstallKey, false);

        string cleanupScript = Path.Combine(Path.GetTempPath(), "uninstall-local-model-lab-" + Guid.NewGuid() + ".cmd");
        File.WriteAllText(
            cleanupScript,
            "@echo off\r\n" +
            "timeout /t 2 /nobreak >nul\r\n" +
            "rmdir /s /q \"" + installDirectory + "\"\r\n" +
            "del /q \"%~f0\"\r\n");
        Process.Start(new ProcessStartInfo("cmd.exe", "/c \"" + cleanupScript + "\"") {
            CreateNoWindow = true,
            UseShellExecute = false,
            WindowStyle = ProcessWindowStyle.Hidden
        });
        return 0;
    }

    private static void StopInstalledProcesses(string installDirectory)
    {
        int currentProcessId = Process.GetCurrentProcess().Id;
        string normalizedRoot = Path.GetFullPath(installDirectory).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        foreach (Process process in Process.GetProcesses())
        {
            if (process.Id == currentProcessId) continue;
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

    private static void DeleteShortcut(string shortcutPath)
    {
        if (File.Exists(shortcutPath)) File.Delete(shortcutPath);
    }
}
